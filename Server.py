import socket as so
from threading import Thread

try:

	num=1
	clients={}
	addr={}
	port=1001
	ss=so.socket()
	ss.bind(("192.168.1.18",port))
	print("Server created")
	ss.listen()
	name={}

	def acc():
		global num
		conn1,ad=ss.accept()
		clients[num]=conn1
		addr[num]=ad
		clients[1].sendall("Enter the number of Participants".encode())
		data=(clients[1].recv(1024)).decode()
		n=int(data)-1
		Thread(target=ri,args=(1,clients[1])).start()
		for i in range(n):
			conn,ad=ss.accept()
			num+=1
			print("connected",num)
			clients[num]=conn
			addr[num]=ad
		for i in range(2,num+1):
			Thread(target=ri,args=(i,clients[num])).start()		

	def ri(n,conn):
		conn.sendall("Enter your name".encode())
		nam=conn.recv(1024).decode()
		name[n]=nam
		print('connection is made with',nam)
		while True:
			rec=conn.recv(1024).decode()
			print(rec)
			se(rec,conn)
		
	def se(rec,conn):
		for f in range(1,num+1):
			r=rec.encode()
			clients[f].sendall(r)

	acc()

except ConnectionResetError:
	print("One of the Clients left the chat")