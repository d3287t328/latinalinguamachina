# Principles: 
Aim to install software at first with flatpaks, toolbox for rare utilities, and for system utils rpm-ostree.

# Primus
Setup private dns (88.198.70.38	88.198.70.39).
rpm-ostree upgrade ; flatpak update ; systemctl reboot 

# Secundus 
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo ;
sudo systemctl disable NetworkManager-wait-online.service ; sudo rm /etc/xdg/autostart/org.gnome.Software.desktop ; systemctl reboot

# Tertius
Install Extension Manager from Software app 

Install user themes, Extension List, App Indicator, Removable Drive Menu, Sound Input, No Overview.

systemctl reboot

# Quartus 

sudo rpm-ostree install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://mirrors.rpmfusion.org/nonfree/fedora/ ; rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm ; systemctl reboot

# Quintus

rpm-ostree install tmux ; rpm-ostree install git ; rpm-ostree install ag ;rpm-ostree install lsd ; rpm-ostree install lsd fzf ; rpm-ostree install neovim ; rpm-ostree install zsh ; sudo usermod --shell /usr/bin/zsh $USER ;  systemctl reboot

# Sextus

Flatpak installs: brave.

cp /var/lib/flatpak/app/com.brave.Browser/active/files/share/applications/com.brave.Browser.desktop ~/.config/autostart ;
cp /usr/share/applications/org.gnome.Shell.desktop ~/.config/autostart ; systemctl restart

# Post install 

oh-my-zsh, bat (https://github.com/sharkdp/bat), .zshrc, hack font, sudo wget https://hosts.ubuntu101.co.za/superhosts.deny -O /etc/hosts.deny/ (https://github.com/Ultimate-Hosts-Blacklist/Ultimate.Hosts.Blacklist), vpn. htop. bandwhich.

# Browser extensions 
ublock origin with updated sources, sponsorblock, video speed controller, Random User-Agent (Switcher), Simple Login, Enhancer for Youtube, Dark Reader.

# Tweak terminal profile preferences

# Reference https://github.com/iaacornus/silverblue-postinstall_upgrade/tree/main