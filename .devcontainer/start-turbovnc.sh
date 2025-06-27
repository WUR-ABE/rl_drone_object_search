#!/bin/bash
echo "starting turbovnc"
sudo rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1
screen -dmS turbovnc bash -c 'VGL_DISPLAY=egl VGL_FPS=30 /opt/TurboVNC/bin/vncserver :1 -depth 24 -noxstartup -securitytypes TLSNone,X509None,None 2>&1 | tee /tmp/vnc.log; read -p "Press any key to continue..."'
# wait for VNC to be running
echo "waiting for turbovnc to be up (display not available messages are expected during this time)"
while ! xdpyinfo -display :1 > /dev/null; do
    sleep 1
done
sleep 1
echo "starting xfce4"
screen -dmS xfce4 bash -c 'DISPLAY=:1 /usr/bin/xfce4-session 2>&1 | tee /tmp/xfce4.log'
echo "starting novnc"
screen -dmS novnc bash -c '/usr/local/novnc/noVNC-1.4.0/utils/novnc_proxy --vnc localhost:5901 --listen 5801 2>&1 | tee /tmp/novnc.log'
