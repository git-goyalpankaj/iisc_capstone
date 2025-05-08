# create_chatbot_structure.ps1

# Root folder (current directory)
$root = Get-Location

# Define all folders relative to $root
$folders = @(
    "app\orchestrator"
    "app\models\schemas"
    "app\rag_engine"
    "app\tools_api\api_clients"
    "app\llm"
    "app\memory"
    "app\observability"
    "app\config"
    "templates"
    "tests\orchestrator"
    "tests\models"
    "tests\rag_engine"
    "tests\tools_api"
    "tests\llm"
    "tests\memory"
    "tests\observability"
    "scripts"
)

# Create folders
foreach ($f in $folders) {
    $path = Join-Path $root $f
    if (-not (Test-Path $path)) {
        New-Item -Path $path -ItemType Directory | Out-Null
        Write-Host "Created folder: $f"
    }
    else {
        Write-Host "Exists: $f"
    }
}

# Create placeholder files
$files = @(
    "app\orchestrator\chatbot_engine.py",
    "app\orchestrator\conversation_manager.py",
    "app\orchestrator\query_enhancer.py",
    "app\orchestrator\prompt_manager.py",
    "app\models\intent_model.py",
    "app\models\sentiment_model.py",
    "app\models\schemas\conversation.py",
    "app\models\schemas\intent.py",
    "app\models\schemas\sentiment.py",
    "app\models\schemas\api_result.py",
    "app\rag_engine\loaders.py",
    "app\rag_engine\index_builder.py",
    "app\rag_engine\retriever.py",
    "app\tools_api\api_caller.py",
    "app\tools_api\api_clients\booking_client.py",
    "app\tools_api\api_clients\cancellation_client.py",
    "app\llm\llm_wrapper.py",
    "app\llm\prompt_templates.py",
    "app\memory\memory_store.py",
    "app\memory\memory_objects.py",
    "app\observability\logger.py",
    "app\observability\metrics.py",
    "app\observability\tracing.py",
    "app\config\settings.py",
    "templates\basic_prompt_template.txt",
    "templates\action_prompt_template.txt",
    "templates\fallback_prompt_template.txt",
    "scripts\build_index.py",
    "scripts\seed_memory.py",
    "scripts\test_api_clients.py",
    "README.md",
    "requirements.txt",
    "pyproject.toml"
)

foreach ($file in $files) {
    $full = Join-Path $root $file
    if (-not (Test-Path $full)) {
        New-Item -Path $full -ItemType File | Out-Null
        Write-Host "Created file: $file"
    }
    else {
        Write-Host "Exists: $file"
    }
}

Write-Host "Folder structure creation complete!"
