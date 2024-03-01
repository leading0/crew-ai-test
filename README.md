# crew-ai-test

# ollama
todo: how to install

## custom model file
```
echo 'from mistral:instruct
PARAMETER stop "Observations:' \
> ollama_instruct_crewai.modelfile

ollama create mistral_crewai -f ollama_instruct_crewai.modelfile
```

# qdrant
```
# linux, asuming directory "qdrant_storage"
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
    
# windows, assuming directory "storage"
docker.exe run -p 6333:6333 -p 6334:6334 -v %cd%\storage:/qdrant/storage:z qdrant/qdrant
```