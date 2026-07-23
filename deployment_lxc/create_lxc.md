# Proxmox setup
Download the Debian 12 template if you don't have it yet:

bash

```bash
pveam update
pveam available | grep debian-12
pveam download local debian-12-standard_12.12-1_amd64.tar.zst
```

Create the container:

bash

```bash
pct create {numeric ID} local:vztmpl/debian-12-standard_12.12-1_amd64.tar.zst \
  --hostname {hostname} \
  --unprivileged 1 \
  --cores 1 \
  --memory 512 \
  --swap 512 \
  --rootfs local-lvm:8 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --features nesting=0
