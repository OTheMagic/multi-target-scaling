param(
    [ValidateSet('auto', 'pdflatex', 'tectonic')]
    [string]$Engine = 'auto',
    [string]$TectonicPath = 'tectonic'
)

$ErrorActionPreference = 'Stop'
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

Push-Location $PSScriptRoot
try {
    if ($Engine -eq 'auto') {
        if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
            $Engine = 'pdflatex'
        } elseif (Get-Command $TectonicPath -ErrorAction SilentlyContinue) {
            $Engine = 'tectonic'
        } else {
            throw 'Install TeX Live/MiKTeX, or pass -Engine tectonic -TectonicPath <executable>.'
        }
    }
    if ($Engine -eq 'pdflatex') {
        Invoke-Checked 'pdflatex' @('-interaction=nonstopmode', '-halt-on-error', 'main.tex')
        Invoke-Checked 'bibtex' @('main')
        Invoke-Checked 'pdflatex' @('-interaction=nonstopmode', '-halt-on-error', 'main.tex')
        Invoke-Checked 'pdflatex' @('-interaction=nonstopmode', '-halt-on-error', 'main.tex')
        foreach ($document in @('response_to_reviewers.tex', 'revision_cover.tex')) {
            Invoke-Checked 'pdflatex' @('-interaction=nonstopmode', '-halt-on-error', $document)
            Invoke-Checked 'pdflatex' @('-interaction=nonstopmode', '-halt-on-error', $document)
        }
    } else {
        Invoke-Checked $TectonicPath @('--keep-logs', '--keep-intermediates', 'compile_main.tex')
        foreach ($extension in @('pdf', 'aux', 'bbl', 'blg', 'log', 'out')) {
            Copy-Item -LiteralPath "compile_main.$extension" -Destination "main.$extension"
        }
        foreach ($document in @('response_to_reviewers.tex', 'revision_cover.tex')) {
            Invoke-Checked $TectonicPath @('--keep-logs', '--keep-intermediates', $document)
        }
    }
    Write-Host 'Built main.pdf, response_to_reviewers.pdf, and revision_cover.pdf.'
} finally {
    Pop-Location
}
