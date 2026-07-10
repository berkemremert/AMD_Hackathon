$ErrorActionPreference = "Stop"

$streamlink = Join-Path $env:APPDATA "Python\Python312\Scripts\streamlink.exe"
$vlc = "C:\Program Files\VideoLAN\VLC\vlc.exe"
$url = "https://www.twitch.tv/lablabai"

if (-not (Test-Path -LiteralPath $streamlink)) {
    throw "Streamlink was not found at $streamlink"
}

if (-not (Test-Path -LiteralPath $vlc)) {
    throw "VLC was not found at $vlc"
}

& $streamlink --player $vlc $url "720p,best"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
