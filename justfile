jupyter_port := "9998"
compute_node := "puck1"
current_dir := justfile_directory()
COML_CLUSTER := "oberon2"
scratch2_deploy_folder := "/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation"

hostname := `hostname`
COML_WORKSPACE := if hostname == "MacBook-Pro-de-jliu" {
    "workspace/RAG/target_neuron_ablation"
} else if hostname == "other-person" {
    "projects/RAG/target_neuron_ablation"
} else {
    "code/RAG/target_neuron_ablation"
}

_default:
  @just --choose

[doc("Open SSH tunnel for remote notebook server.")]
notebook-tunnel node=compute_node port=jupyter_port:
    @echo "Creating a tunnel to {{node}}:{{port}}"
    ssh -L "{{port}}:{{node}}:{{port}}" "{{node}}" -N


[doc("Fetch notebooks from Oberon")]
fetch-notebooks:
    echo "Fetching notebooks..."
    rsync -azP --delete --exclude=".venv" --exclude=".ipynb_checkpoints" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/notebooks/" "{{current_dir}}/notebooks/"


[doc("Deploy source code to remote")]
deploy-coml: exec-permissions
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".venv" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{COML_CLUSTER}}:{{scratch2_deploy_folder}}"

[doc("Make executables")]
exec-permissions:
    find src/scripts -name "*.py" -exec chmod +x {} \;
    find experiments -name "*.sh" -exec chmod +x {} \;

[doc("Install module & dependencies")]
install:
    pip install -e ".[dev]"
    mypy --install-types

[doc("Run Jupyter Server Locally")]
run-notebook:
    jupyter lab

[doc("Check Syntax (RUFF)")]
syntax-check:
    ruff check

[doc("Check Typing (mypy)")]
type-check:
    mypy target_neuron_ablation

[doc("Auto Formatting (RUFF)")]
format:
    ruff format target_neuron_ablation

[doc("Commit and push all changes")]
add-commit-push m="":
    # git add .
    @[[ ! -z "{{m}}" ]] &&  echo "commiting:: {{m}}" # git commit -m "{{m}}"
    @[[ -z "{{m}}" ]] &&  echo "commiting:: empty" # git commit -m "{{m}}"
    # git push

check-todo:
    @rg \
    --glob !notebooks/ \
    --glob !justfile \
    --glob !pyproject.toml \
    --ignore-case \
    'fixme|todo|feat' \
    .