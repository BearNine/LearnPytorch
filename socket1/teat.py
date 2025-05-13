import socket

print("asds")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("192.168.0.120",1234))
    s.listen()
    c, addr = s.accept()
    while(1):
        with c:
            print(addr, "connected.")

            while True:
                data = c.recv(1024)
                c.sendall(data)