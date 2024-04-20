+++
title = 'SFTP guide'
date = 2022-08-15T10:46:52+08:00
draft = false
math = true
tags = ['sftp', 'network']
categories = ['code']
summary = "Simple guide to use SFTP in terminal"
+++

## Preface

由于最近需要在服务器上进行操作，所以常有在本地和服务器上进行文件传输的需求。因此装了 SFTP for VS code，也熟悉了一下 SFTP 的指令。可以说，这个工具真的是非常的便捷。这篇文章就简单介绍一下常用的 SFTP 指令，参考了 [How To Use SFTP to Securely Transfer Files with a Remote Server](https://www.digitalocean.com/community/tutorials/how-to-use-sftp-to-securely-transfer-files-with-a-remote-server) 这篇文章。



## Connect with SFTP

SFTP 默认使用 SSH 协议来建立远程连接。SSH 的连接命令为：

```
$ ssh user_name@ip
```

退出命令为：

```
$ exit
```

同样地，SFTP 的命令为：

```
$ sftp user_name@ip
```

默认连接到 port 22，当然也可以调整端口，通过`-oPort`命令。请自行搜索。



## Navigate with SFTP

连接之后，终端会出现`sftp> `标识。我们可以使用熟悉的 Linux 指令来进行操作。例如

```
sftp> pwd
sftp> ls
sftp> cd ..
```

这些指令是在远程服务器上的操作。

如果想在本地文件系统操作，需要在这些指令的前面加上一个 `l`。

```
sftp> lpwd

Output
Local working directory: /Users/...
```

```
sftp> lls
```

```
sftp> lcd Desktop
```

遇到其它问题，可以使用`help`或者`?`指令来查看帮助。



## Transfer Files with SFTP

如果想要从服务器上下载文件，可以使用`get`命令。

```
sftp> get <filename>
```

这个指令把该文件下载到本地系统中，并给它相同的命名。如果想要改名，方法为：

```
sftp> get <remoteFilename> <localFilename>
```

如果想要下载整个文件夹，可以加上`-r`：

```
sftp> get -r <directoryName>
```



如果需要将本地文件上传至服务器，可以使用`put`指令。

```
sftp> put <filename>
```

其它操作类似`get`，请自行尝试。



可以使用`!`来切回到本地的终端，例如 Git Bash。

```
sftp> !
```

此后就能使用本地终端上的指令啦。如果想要回到 SFTP，可以使用`exit`指令：

```
$ exit
```



## Else

SFTP 还有其它强大的功能，例如修改文件的权限、用户的分组等。`chmod`、`rmdir` 等指令都是可以使用的——当然，如果直接输入这些指令，操作都是在服务器上进行的；如果想要在本地文件系统操作，需要在所有指令前面加上`!`。例如：

```
sftp> !chmod 644 <filename>
```

