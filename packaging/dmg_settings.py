# dmgbuild settings for the macOS disk image.
# Usage: dmgbuild -s packaging/dmg_settings.py "Lyra" dist/Lyra.dmg
import os.path

app_name = "Lyra"
application = os.path.abspath(f"dist/{app_name}.app")

# Contents of the DMG: the app plus a symlink to /Applications for drag-install.
files = [application]
symlinks = {"Applications": "/Applications"}

# Layout
window_rect = ((200, 200), (520, 360))
icon_locations = {f"{app_name}.app": (130, 170), "Applications": (390, 170)}
default_view = "icon-view"
icon_size = 96
