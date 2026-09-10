param([ValidateSet('stage','restore')][string]$Mode = 'stage')

$ErrorActionPreference = 'Stop'
$workspacePath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')).TrimEnd('\')
$planPath = Join-Path $PSScriptRoot 'move_plan.json'
$cleanupPlan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
$stagingPath = [System.IO.Path]::GetFullPath([string]$cleanupPlan.staging_root).TrimEnd('\')
if ($workspacePath -ne 'E:\multi-target-scaling') { throw "Unexpected workspace: $workspacePath" }
if ([System.IO.Path]::GetFullPath([string]$cleanupPlan.workspace).TrimEnd('\') -ne $workspacePath) { throw 'Plan workspace differs.' }
if ($stagingPath -ne (Join-Path $workspacePath 'deletable\cleanup_2026-09-09')) { throw 'Unexpected staging root.' }

function Assert-WorkspaceFile([string]$candidate) {
    $absolute = [System.IO.Path]::GetFullPath($candidate)
    if (-not $absolute.StartsWith($workspacePath + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes workspace: $absolute"
    }
    foreach ($forbidden in @('.git','.codex','.agents')) {
        $blocked = Join-Path $workspacePath $forbidden
        if ($absolute -eq $blocked -or $absolute.StartsWith($blocked + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Protected internal path: $absolute"
        }
    }
    return $absolute
}

# Resolve and verify EVERY source/destination before making any move. No wildcard moves.
$operations = @()
foreach ($record in $cleanupPlan.records) {
    $original = Assert-WorkspaceFile ([string]$record.source)
    $staged = Assert-WorkspaceFile ([string]$record.destination)
    $expectedOriginal = [System.IO.Path]::GetFullPath((Join-Path $workspacePath ([string]$record.path)))
    $expectedStaged = [System.IO.Path]::GetFullPath((Join-Path $stagingPath ([string]$record.path)))
    if ($original -ne $expectedOriginal -or $staged -ne $expectedStaged) { throw 'Manifest path mismatch.' }
    if (-not $staged.StartsWith($stagingPath + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Staging escape.' }
    if ($Mode -eq 'stage') { $from = $original; $to = $staged } else { $from = $staged; $to = $original }
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { throw "Missing source: $from" }
    if (Test-Path -LiteralPath $to) { throw "Refusing to overwrite: $to" }
    $item = Get-Item -LiteralPath $from -Force
    if ($item.Length -ne [long]$record.bytes) { throw "Size changed: $from" }
    $ancestor = $item
    while ($ancestor.FullName -ne $workspacePath) {
        if (($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point: $from" }
        if ($ancestor -is [System.IO.FileInfo]) { $ancestor = $ancestor.Directory } else { $ancestor = $ancestor.Parent }
        if ($null -eq $ancestor) { throw 'Ancestor escaped workspace.' }
    }
    $hash = (Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne [string]$record.sha256) { throw "Content changed: $from" }
    $operations += [pscustomobject]@{ From=$from; To=$to; Record=$record }
}

$journalPath = Join-Path $PSScriptRoot ($Mode + '_journal.jsonl')
if (Test-Path -LiteralPath $journalPath) { throw "Existing operation journal: $journalPath. Inspect before retrying." }
$movedFiles = 0
$movedBytes = [long]0
foreach ($operation in $operations) {
    $parentPath = Split-Path -Parent $operation.To
    if (-not (Test-Path -LiteralPath $parentPath)) {
        New-Item -ItemType Directory -Path $parentPath -Force | Out-Null
    }
    Move-Item -LiteralPath $operation.From -Destination $operation.To -ErrorAction Stop
    $actualHash = (Get-FileHash -LiteralPath $operation.To -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne [string]$operation.Record.sha256) { throw "Post-move hash mismatch: $($operation.To)" }
    [pscustomobject]@{ path=$operation.Record.path; source=$operation.From; destination=$operation.To;
        bytes=$operation.Record.bytes; sha256=$actualHash; moved_utc=[DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json -Compress | Add-Content -LiteralPath $journalPath -Encoding UTF8
    $movedFiles += 1
    $movedBytes += [long]$operation.Record.bytes
}
$result = [pscustomobject]@{ status='complete'; operation=$Mode; files=$movedFiles; bytes=$movedBytes;
    staging_root=$stagingPath; hashes_verified=$true; deleted_files=0; completed_utc=[DateTime]::UtcNow.ToString('o') }
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot ($Mode + '_result.json')) -Encoding UTF8
$result | ConvertTo-Json
