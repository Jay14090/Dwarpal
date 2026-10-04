<#
Publish the laptop webcam to MediaMTX as rtsp://localhost:8554/webcam (run on WINDOWS, not WSL).
WSL2 cannot see USB webcams directly, so Windows captures (DirectShow) and the Dwarpal worker
in WSL2 reads the RTSP stream. Requires ffmpeg on Windows:  winget install Gyan.FFmpeg

Usage (PowerShell):
  .\scripts\webcam_publish.ps1                       # first camera, device default mode
  .\scripts\webcam_publish.ps1 -List                 # list cameras
  .\scripts\webcam_publish.ps1 -Device "Integrated Camera" -Size 1280x720 -Fps 15
#>
param(
  [string]$Device = "",
  [string]$Url = "rtsp://localhost:8554/webcam",
  [string]$Size = "",
  [int]$Fps = 15,
  [switch]$List
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Write-Error "ffmpeg not found. Install with: winget install Gyan.FFmpeg (then reopen the terminal)"
}

function Get-VideoDevices {
  $out = & ffmpeg -hide_banner -list_devices true -f dshow -i dummy 2>&1 | Out-String
  [regex]::Matches($out, '"([^"]+)"\s+\(video\)') | ForEach-Object { $_.Groups[1].Value }
}

$devices = @(Get-VideoDevices)
if ($List) { $devices | ForEach-Object { Write-Output $_ }; exit 0 }
if (-not $Device) {
  if ($devices.Count -eq 0) { Write-Error "no DirectShow video devices found" }
  $Device = $devices[0]
}
Write-Output "Publishing '$Device' -> $Url  (Ctrl+C to stop)"

$inputArgs = @("-f", "dshow", "-rtbufsize", "64M", "-framerate", "$Fps")
if ($Size) { $inputArgs += @("-video_size", $Size) }
$inputArgs += @("-i", "video=$Device")
$encodeArgs = @("-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
  "-g", "$Fps", "-bf", "0", "-pix_fmt", "yuv420p", "-an",
  "-f", "rtsp", "-rtsp_transport", "tcp", $Url)
& ffmpeg -hide_banner -loglevel warning @inputArgs @encodeArgs
