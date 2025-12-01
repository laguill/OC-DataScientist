# Prérequis

Utilisation de pyspark 3.1.2
java 11 doit etre installé sur le système

```bash
# Installer Java 11
sudo zypper install java-11-openjdk java-11-openjdk-devel

# Vérifier que Java 11 est installé
sudo update-alternatives --config java
```

Vous devriez voir une liste comme :
```
  Selection    Path                                          Priority
------------------------------------------------------------
* 0            /usr/lib64/jvm/java-21-openjdk-21/bin/java     2101
  1            /usr/lib64/jvm/java-11-openjdk-11/bin/java     1101
```
