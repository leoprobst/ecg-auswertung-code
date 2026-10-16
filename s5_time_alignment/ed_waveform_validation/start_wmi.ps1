param(
    [Parameter(Mandatory = $true)][string]$CommandLine,
    [string]$Cwd = (Get-Location).Path
)
# Startet einen vom Aufrufer unabhaengigen Prozess ueber WMI (Win32_Process.Create) und gibt die ProcessId (cmd.exe) aus.
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $CommandLine; CurrentDirectory = $Cwd }
if ($r.ReturnValue -ne 0) { Write-Error "WMI-Create fehlgeschlagen: $($r.ReturnValue)"; exit 1 }
Write-Output $r.ProcessId
