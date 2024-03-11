import stat
import paramiko
import socket
from pathlib import Path
import click

@click.command()
@click.option('--proxy-host', default='proxy.uni-rostock.de', help='Proxy host name.')
@click.option('--proxy-port', type=int, default=8080, help='Proxy port number.')
@click.option('--ssh-host', default="agingkills.eu", help='SSH server host name.')
@click.option('--ssh-port', type=int, default=22, help='SSH server port number.')
@click.option('--ssh-username', default="antonkulaga", help='SSH username.')
@click.option('--ssh-password', required=True, help='SSH password.')
@click.option('--remote-directory', default="/data/papers/s2orc/processed_papers", help='Remote directory to download from.')
@click.option('--local-directory', default="/data/uq2347/processed_papers", help='Local directory to download to.')
def main(proxy_host: str, proxy_port: int, ssh_host: str, ssh_port: int, ssh_username: str, ssh_password: str, remote_directory: str, local_directory: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((proxy_host, proxy_port))

    connect_command = f"CONNECT {ssh_host}:{ssh_port} HTTP/1.1\r\nHost: {ssh_host}\r\n\r\n"
    sock.sendall(connect_command.encode())
    sock.recv(4096)  # In a real application, you should parse this response.

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    transport = paramiko.Transport(sock)
    transport.connect(username=ssh_username, password=ssh_password)

    sftp = paramiko.SFTPClient.from_transport(transport)
    download_directory(sftp, Path(remote_directory), Path(local_directory))

    sftp.close()
    transport.close()
    sock.close()

def download_directory(sftp: paramiko.SFTPClient, remote_dir: Path, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(str(remote_dir)):
        remote_item_path = remote_dir / item.filename
        local_item_path = local_dir / item.filename

        if stat.S_ISDIR(item.st_mode):  # If item is a directory, recurse into it
            download_directory(sftp, remote_item_path, local_item_path)
        else:
            if not local_item_path.exists():  # Only download if the file doesn't exist
                print(f"Downloading {remote_item_path} to {local_item_path}")
                sftp.get(str(remote_item_path), str(local_item_path))
            else:
                print(f"Skipping {remote_item_path} as it already exists.")

if __name__ == '__main__':
    main()
