//! Collect public hardware facts only: no hostname, user name, serials, or paths.
use super::MachineInfo;

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn command(program: &str, args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "{program} could not collect machine information: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8(out.stdout)
        .map(|s| s.trim().trim_start_matches('\u{feff}').to_owned())
        .map_err(|e| e.to_string())
}

#[cfg(target_os = "windows")]
pub fn collect() -> Result<MachineInfo, String> {
    // UTF-8 output also handles non-English Windows installations.
    let script = r#"
        $ErrorActionPreference = 'Stop'
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
        Add-Type -AssemblyName Microsoft.VisualBasic
        $benchCpu = Get-ItemProperty 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0'
        $benchMemory = [Microsoft.VisualBasic.Devices.ComputerInfo]::new().TotalPhysicalMemory
        try {
            $benchDevices = @(Get-CimInstance Win32_PnPSignedDriver |
                Where-Object { $_.DeviceClass -eq 'DISPLAY' -or $_.DeviceName -match '\b(NPU|IPU)\b|Neural|Ryzen AI' } |
                Sort-Object DeviceName -Unique |
                ForEach-Object { @{ name = $_.DeviceName; driver_version = $_.DriverVersion } })
        } catch {
            # Registry fallback when WMI/CIM is restricted. Missing devices are
            # omitted; actual runtime device/driver metadata is still required.
            $benchDevices = @(Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class' -ErrorAction SilentlyContinue |
                ForEach-Object { Get-ChildItem $_.PSPath -ErrorAction SilentlyContinue } |
                ForEach-Object { Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue } |
                Where-Object { $_.DriverDesc -and ($_.PSParentPath -like '*{4d36e968-e325-11ce-bfc1-08002be10318}*' -or $_.DriverDesc -match '\b(NPU|IPU)\b|Neural|Ryzen AI') } |
                Sort-Object DriverDesc -Unique |
                ForEach-Object { @{ name = $_.DriverDesc; driver_version = $_.DriverVersion } })
        }
        $benchPower = (powercfg /getactivescheme | Out-String).Trim()
        @{
            cpu = $benchCpu.ProcessorNameString.Trim()
            logical_cpus = [System.Environment]::ProcessorCount
            memory_bytes = [uint64]$benchMemory
            os = [System.Environment]::OSVersion.VersionString
            architecture = $env:PROCESSOR_ARCHITECTURE
            accelerators = $benchDevices
            power_profile = $benchPower
        } | ConvertTo-Json -Depth 4 -Compress
    "#;
    let data = command(
        "powershell.exe",
        &["-NoProfile", "-NonInteractive", "-Command", script],
    )?;
    serde_json::from_str(&data).map_err(|e| format!("reading Windows machine information: {e}"))
}

#[cfg(target_os = "linux")]
pub fn collect() -> Result<MachineInfo, String> {
    let cpu = std::fs::read_to_string("/proc/cpuinfo").map_err(|e| e.to_string())?;
    let memory = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    let os = std::fs::read_to_string("/etc/os-release").map_err(|e| e.to_string())?;
    let cpu = cpu
        .lines()
        .find_map(|line| {
            line.strip_prefix("model name")?
                .split_once(':')
                .map(|(_, v)| v.trim().to_owned())
        })
        .ok_or("CPU name unavailable; supply machine metadata explicitly")?;
    let memory_kib: u64 = memory
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemTotal:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
        .ok_or("RAM size unavailable")?;
    let os = os
        .lines()
        .find_map(|line| line.strip_prefix("PRETTY_NAME="))
        .unwrap_or("Linux")
        .trim_matches('"')
        .to_owned();
    Ok(MachineInfo {
        cpu,
        logical_cpus: std::thread::available_parallelism()
            .map_err(|e| e.to_string())?
            .get(),
        memory_bytes: memory_kib * 1024,
        os,
        architecture: std::env::consts::ARCH.into(),
        accelerators: vec![],
        power_profile: None,
    })
}

#[cfg(target_os = "macos")]
pub fn collect() -> Result<MachineInfo, String> {
    Ok(MachineInfo {
        cpu: command("sysctl", &["-n", "machdep.cpu.brand_string"])?,
        logical_cpus: command("sysctl", &["-n", "hw.logicalcpu"])?
            .parse::<usize>()
            .map_err(|e| e.to_string())?,
        memory_bytes: command("sysctl", &["-n", "hw.memsize"])?
            .parse::<u64>()
            .map_err(|e| e.to_string())?,
        os: command("sw_vers", &[])?,
        architecture: std::env::consts::ARCH.into(),
        accelerators: vec![],
        power_profile: None,
    })
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub fn collect() -> Result<MachineInfo, String> {
    Err("automatic machine information is unavailable on this OS".into())
}
