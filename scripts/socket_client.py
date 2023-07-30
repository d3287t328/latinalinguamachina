import socket
 
def Main():
    host = "localhost"
    port = 51495

    mySocket = socket.socket()
    mySocket.bind((host,port))

    mySocket.listen(1)
    conn, addr = mySocket.accept()
    print(f"Connection from: {str(addr)}")
    while True:
        data = conn.recv(1024).decode()
        if not data:
                break
        print(f"from connected  user: {str(data)}")

        data = str(data).upper()
        print(f"sending: {data}")
        conn.send(data.encode())

    conn.close()
     
if __name__ == '__main__':
    Main()