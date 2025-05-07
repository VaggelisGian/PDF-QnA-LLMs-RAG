try {
    $apiUrl = "http://localhost:8000/api/batch_chat"
    $requestBodyPath = "batch_request.json"

    $requestBody = Get-Content -Raw -Path $requestBodyPath -ErrorAction Stop

    Write-Host "Sending request to $apiUrl..."
    $response = Invoke-RestMethod -Uri $apiUrl -Method Post -ContentType 'application/json' -Body $requestBody -TimeoutSec 180 -ErrorAction Stop

    if ($null -ne $response -and $null -ne $response.results) {
        $transformedQuestions = $response.results | ForEach-Object {
            [PSCustomObject]@{
                question  = $_.question
                use_graph = $_.use_graph
                answer    = if ($null -eq $_.answer) { "" } else { $_.answer }
            }
        }

        $finalOutput = [PSCustomObject]@{
            questions = $transformedQuestions
        }

        $outputFilePath = Join-Path -Path (Get-Location) -ChildPath "results.json"

        $finalOutput | ConvertTo-Json -Depth 5 | Out-File -FilePath $outputFilePath -Encoding utf8
        Write-Host "Results successfully saved to $outputFilePath"
    }
    else {
        Write-Error "Failed to get a valid response or the response did not contain 'results' array."
        if ($null -ne $response) {
            Write-Host "Raw response from server:"
            $response | ConvertTo-Json -Depth 5 | Write-Host
        }
    }
}
catch {
    Write-Error "An error occurred: $($_.Exception.Message)"
    Write-Error "Failed to process the request and save results."
    if ($_.Exception.Response) {
        $errorResponseStream = $_.Exception.Response.GetResponseStream()
        $streamReader = New-Object System.IO.StreamReader($errorResponseStream)
        $errorBody = $streamReader.ReadToEnd()
        $streamReader.Close()
        $errorResponseStream.Close()
        Write-Host "Error response body from server:"
        Write-Host $errorBody
    }
}