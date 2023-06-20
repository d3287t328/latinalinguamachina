# Principles: 
Aim to install software at first with flatpaks, toolbox for rare utilities, and for system utils rpm-ostree.

# Primus
Setup private dns:

add the dns line to the ipv4 section.
sudo nano /etc/NetworkManager/system-connections/enp0s3.nmconnection 
dns=88.198.70.38, 88.198.70.39

sudo sed -i 's/^AutomaticUpdatePolicy =.*/AutomaticUpdatePolicy = check/' /etc/rpm-ostreed.conf ; rpm-ostree reload
 
systemctl enable rpm-ostreed-automatic.timer --now ; rpm-ostree upgrade ; flatpak update

systemctl reboot 

# Secundus 
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo ;
flatpak remote-add --if-not-exists fedora oci+https://registry.fedoraproject.org ; 
flatpak update --appstream ; flatpak update ; 
sudo systemctl disable NetworkManager-wait-online.service ; sudo rm /etc/xdg/autostart/org.gnome.Software.desktop 

sudo rpm-ostree install akmod-nvidia xorg-x11-drv-nvidia-cuda
systemctl reboot

# Tertius
Install Extension Manager from Software app 

Install user themes, Extension List, App Indicator, Removable Drive Menu, Sound Input, No Overview.

# Quartus 

sudo rpm-ostree install ttps://mirrors.rpmfusion.org/nonfree/fedora/ ;
sudo rpm-ostree install bat lsd fzf git tmux ag cronie pip ; systemctl reboot

# Sextus

Flatpak installs: brave, neovim
(crontab -l ; echo "0 22 * * 0 sudo rpm-ostree upgrade && sudo systemctl reboot") | crontab -  [ignore any errors]
cp /var/lib/flatpak/app/com.brave.Browser/active/files/share/applications/com.brave.Browser.desktop ~/.config/autostart ;
cp /usr/share/applications/org.gnome.Shell.desktop ~/.config/autostart ; systemctl restart

# Post install 

sudo mkdir /etc/host.deny ; wget https://hosts.ubuntu101.co.za/superhosts.deny -O /etc/hosts.deny [https://github.com/Ultimate-Hosts-Blacklist/Ultimate.Hosts.Blacklist], vpn. htop.

# Useful browser extensions 
ublock origin with updated sources, sponsorblock, video speed controller, Random User-Agent (Switcher), Simple Login, Enhancer for Youtube, Dark Reader.

# Tweak terminal profile preferences

# Reference https://github.com/iaacornus/silverblue-postinstall_upgrade/tree/main
            https://fedoramagazine.org/how-i-customize-fedora-silverblue-and-fedora-kinoite/
