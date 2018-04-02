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

if [ ! -z ${HELP} ] || [ -z ${POSITIONAL[0]} ] || [ -z ${POSITIONAL[1]} ]
then
	echo "Usage : $0 [-h|--help] url/ hash"
	echo "Url like http://adresse/path/.git/"
	exit
fi

url="${POSITIONAL[0]}"
hash="${POSITIONAL[1]}"

# download the file
HTTP_STATUS=$(curl --silent -w "%{http_code}" "${url}objects/${hash:0:2}/${hash:2}" --create-dirs -o "${hash}" 2>&1)

# if the request went ok
if [ ${HTTP_STATUS} == "200" ]
then
	echo -e "\033[32m[+] ${hash} retrieved\033[0m"
else
	# if the request went not ok, rm it 
	rm "${hash}"
	echo -e "\033[31m[-] Could not retrieve ${hash}\033[0m"
fi