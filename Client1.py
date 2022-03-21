from tkinter import *
import socket as so
from threading import Thread

try:
	port=1001
	cs=so.socket()
	cs.connect(("192.168.1.18",port))

	def rect():
		while True:
			data=cs.recv(1024).decode()
			mess.insert(END,data)

	def send():
		user_input=txt.get()
		cs.sendall(user_input.encode())
		mestxt.set("")

	def qui():
		cs.close()
		window.quit()

	window=Tk()
	window.title("Client")
	window.geometry("960x660")
	mestxt=StringVar()
	mestxt.set("Enter your text Here")
	txt=Entry(window,textvariable=mestxt)
	sen=Button(window,text="Send",command=send)
	quit=Button(window,text="Quit",command=qui)
	scr=Scrollbar()
	mess=Listbox(window,yscrollcommand=scr.set,font=("Helvetica","18"))
	scr=Scrollbar(mess,orient="vertical")
	scr.config(command=mess.yview)
	txt.place(height=40,width=780,x=0,y=620)
	sen.place(height=40,width=80,x=780,y=620)
	quit.place(height=40,width=80,x=860,y=620) 
	scr.pack(side=RIGHT,fill=Y)
	mess.place(height=620,width=950,x=0,y=0)

	window.protocol("WM_DELETE_WINDOW",qui)

	receiveing_thread=Thread(target=rect)
	receiveing_thread.start()
	window.mainloop()
	
except ConnectionAbortedError:
	print("Please resart the programme. Connection to server is lost")
	cs.close()

except ConnectionRefusedError:
	print("Check if the Server is running. Please restart the programme")

except ConnectionError:
	print("There is a problem in the connection. Please restart the programme")

except Exception:
	print("There was a unidentified Error. Please restart the programme")
	cs.close()