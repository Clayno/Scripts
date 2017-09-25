import binascii
import argparse
import hashlib
import os
import re
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument("file", help="Contenant un message snmpv3", type=str)
parser.add_argument("wordlist", help="Mots utilisés pour le brute force", type=str)
parser.add_argument("thread", help="Nombre de thread pour le brute force", type=int)
parser.parse_args()
args = parser.parse_args()

with open(args.wordlist, "r", encoding='ISO-8859-1') as f:
    contenu = f.read()
f.close()
listPass = contenu.split("\n")

with open(args.file, "r") as f:
    contenu = f.read()
f.close()
whole_message = binascii.unhexlify(contenu)
print("[+] Parametres")
print("")


IPAD = b"\x36"*64
OPAD = b"\x5c"*64
msgAuthoritativeEngineID = whole_message[31:47]
strMsgAuthoritativeEngineID = binascii.hexlify(msgAuthoritativeEngineID)
print("msgAuthoritativeEngineID : ", strMsgAuthoritativeEngineID)
msgAuthenticationParameters = whole_message[62:74]
tmp2 = binascii.hexlify(msgAuthenticationParameters)
print("msgAuthenticationParameters : ", tmp2)
success = False
print("Taille de la wordlist : ", len(listPass))

def processus(password) :
	if password != "" :	
		print(password)
		auth_key = os.popen("snmpkey md5 '%s' %s" % (password, strMsgAuthoritativeEngineID.decode())).read()
		search = re.search(r'(?<=0x)\w{32}', auth_key)
		if search != None :
			auth_key = search.group(0)
		else :
			print("Erreur : ", auth_key)
		extendedAuthKey = binascii.unhexlify(auth_key + "00"*48)
		K1 = "".join([chr(x ^ y) for x,y in zip(extendedAuthKey, IPAD)]).encode()
		K2 = "".join([chr(x ^ y) for x,y in zip(extendedAuthKey, OPAD)]).encode()
		hash_k1_msg = binascii.unhexlify(hashlib.md5(K1 + whole_message).hexdigest())
		final_hash = hashlib.md5(K2 + hash_k1_msg).hexdigest()[:12]
		if final_hash == msgAuthenticationParameters :
			print("Yaaaaay ! Mot de passe : ", password)
			success = True
		
if __name__ == '__main__':	
	pool = Pool(args.thread)
	pool.map(processus, listPass)
	if not success :
		print("Erf... Pas marche....")