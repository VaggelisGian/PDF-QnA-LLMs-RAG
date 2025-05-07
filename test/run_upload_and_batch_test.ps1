param (
    [Parameter(ParameterSetName = 'UploadAndTest', Mandatory = $true)]
    [string]$PdfFilePath,

    [Parameter(ParameterSetName = 'TestOnly', Mandatory = $true)]
    [string]$DocumentTitleToQuery,

    [Parameter(Mandatory = $false)]
    [string]$QuestionsJsonPath
)

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $PSBoundParameters.ContainsKey('QuestionsJsonPath')) {
    $QuestionsJsonPath = Join-Path -Path $ScriptDirectory -ChildPath "batch_questions_template.json"
}

try {
    $fileNameForBatchChat = ""

    if ($PSCmdlet.ParameterSetName -eq 'UploadAndTest') {
        $uploadApiUrl = "http://localhost:8000/api/upload"
        Write-Host "Uploading PDF: $PdfFilePath to $uploadApiUrl..."

        if (-not (Test-Path $PdfFilePath)) {
            Write-Error "PDF file not found at: $PdfFilePath"
            exit 1
        }

        $fileBytes = [System.IO.File]::ReadAllBytes($PdfFilePath)
        $pdfFileNameForUpload = [System.IO.Path]::GetFileName($PdfFilePath)
        $boundary = [System.Guid]::NewGuid().ToString()
        $LF = "`r`n"

        $bodyLines = (
            "--$boundary",
            "Content-Disposition: form-data; name=`"file`"; filename=`"$pdfFileNameForUpload`"",
            "Content-Type: application/octet-stream$LF",
            ([System.Text.Encoding]::GetEncoding("ISO-8859-1").GetString($fileBytes)),
            "--$boundary--$LF"
        ) -join $LF

        $headers = @{
            "Content-Type" = "multipart/form-data; boundary=`"$boundary`""
        }

        try {
            $uploadResponse = Invoke-RestMethod -Uri $uploadApiUrl -Method Post -Body $bodyLines -Headers $headers -ErrorAction Stop
        }
        catch {
            Write-Error "Error during PDF upload: $($_.Exception.Message)"
            if ($_.Exception.Response) {
                $errorResponseStream = $_.Exception.Response.GetResponseStream()
                $streamReader = New-Object System.IO.StreamReader($errorResponseStream)
                $errorBody = $streamReader.ReadToEnd()
                $streamReader.Close()
                $errorResponseStream.Close()
                Write-Host "Error response body from upload server:"
                Write-Host $errorBody
            }
            exit 1
        }

        $fileNameForBatchChat = $uploadResponse.filename
        $jobId = $uploadResponse.job_id
        Write-Host "PDF uploaded successfully. Filename from server: $fileNameForBatchChat, Job ID: $jobId"

        $progressApiUrlBase = "http://localhost:8000/api/progress"
        $maxWaitSeconds = 6000000 
        $pollIntervalSeconds = 10
        $elapsedWaitSeconds = 0
        $processingComplete = $false

        Write-Host "Waiting for document processing to complete (Job ID: $jobId)..."
        while (-not $processingComplete -and $elapsedWaitSeconds -lt $maxWaitSeconds) {
            Start-Sleep -Seconds $pollIntervalSeconds
            $elapsedWaitSeconds += $pollIntervalSeconds
            Write-Host "Polling job status... ($($elapsedWaitSeconds)s / $($maxWaitSeconds)s)"
            try {
                $progressResponse = Invoke-RestMethod -Uri "$progressApiUrlBase/$jobId" -Method Get -ErrorAction SilentlyContinue
                if ($null -ne $progressResponse) {
                    Write-Host "Job Status: $($progressResponse.status), Message: $($progressResponse.message), Percent: $($progressResponse.percent_complete)%"
                    if ($progressResponse.status -eq "completed" -or $progressResponse.status -eq "completed_empty") {
                        $processingComplete = $true
                        Write-Host "Document processing finished successfully."
                    }
                    elseif ($progressResponse.status -eq "failed") {
                        Write-Error "Document processing failed: $($progressResponse.message)"
                        exit 1
                    }
                }
                else {
                    Write-Warning "Failed to get progress for job $jobId. Will retry."
                }
            }
            catch {
                Write-Warning "Error polling job status: $($_.Exception.Message). Will retry."
            }
        }

        if (-not $processingComplete) {
            Write-Error "Document processing timed out after $maxWaitSeconds seconds."
            exit 1
        }
    }
    elseif ($PSCmdlet.ParameterSetName -eq 'TestOnly') {
        Write-Host "Skipping PDF upload and processing. Using existing document title: $DocumentTitleToQuery"
        $fileNameForBatchChat = $DocumentTitleToQuery
    }
    else {
        Write-Error "Invalid parameter set used. Please use 'UploadAndTest' or 'TestOnly'."
        exit 1
    }

    if ([string]::IsNullOrEmpty($fileNameForBatchChat)) {
        Write-Error "The document title for batch chat could not be determined. Exiting."
        exit 1
    }

    $batchChatApiUrl = "http://localhost:8000/api/batch_chat"
    Write-Host "Preparing batch chat request for document: $fileNameForBatchChat"

    if (-not (Test-Path $QuestionsJsonPath)) {
        Write-Error "Questions JSON template file not found at: $QuestionsJsonPath"
        exit 1
    }

    $questionsTemplateContent = Get-Content -Raw -Path $QuestionsJsonPath | ConvertFrom-Json
    if ($null -eq $questionsTemplateContent.questions) {
        Write-Error "The questions template JSON at '$QuestionsJsonPath' must have a 'questions' array."
        exit 1
    }

    $batchRequestBody = @{
        document_title = $fileNameForBatchChat
        questions      = $questionsTemplateContent.questions
    } | ConvertTo-Json -Depth 5

    Write-Host "Sending batch chat request to $batchChatApiUrl..."
    $batchChatResponse = Invoke-RestMethod -Uri $batchChatApiUrl -Method Post -ContentType 'application/json' -Body $batchRequestBody -TimeoutSec 300000 -ErrorAction Stop

    if ($null -ne $batchChatResponse -and $null -ne $batchChatResponse.results) {
        $transformedQuestions = $batchChatResponse.results | ForEach-Object {
            [PSCustomObject]@{
                question  = $_.question
                use_graph = $_.use_graph
                answer    = if ($null -eq $_.answer) { "" } else { $_.answer }
            }
        }
        $finalOutput = @{ questions = $transformedQuestions }

        $outputFilePath = Join-Path -Path $ScriptDirectory -ChildPath "results.json"

        $finalOutput | ConvertTo-Json -Depth 5 | Out-File -FilePath $outputFilePath -Encoding utf8
        Write-Host "Batch chat results successfully saved to $outputFilePath"
    }
    else {
        Write-Error "Failed to get a valid batch chat response or the response did not contain 'results' array."
        if ($null -ne $batchChatResponse) {
            Write-Host "Raw batch chat response from server:"
            $batchChatResponse | ConvertTo-Json -Depth 5 | Write-Host
        }
    }

}
catch {
    Write-Error "An error occurred: $($_.Exception.Message)"
    Write-Error "Script failed."
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