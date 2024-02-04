import json
from typing import Optional
import click
import requests
from pathlib import Path

# Create a context class to hold global options
class Context:
    def __init__(self, api_key: str):
        self.api_key = api_key

# Setup a group with a shared context
@click.group()
@click.option('--api-key', type=str, help='Semantic Scholar API key', required=True)
@click.pass_context
def cli(ctx: click.Context, api_key: str):
    """CLI tool to interact with Semantic Scholar's dataset API."""
    ctx.obj = Context(api_key)

def make_request(url: str, ctx: click.Context) -> dict:
    headers = {
        'x-api-key': ctx.obj.api_key,
        'Accept': 'application/json'  # Explicitly request JSON response
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def write_output(data: dict, output_path: Optional[Path]):
    if output_path:
        output_path.write_text(json.dumps(data, indent=2))
        click.echo(f"Output written to {output_path}")
    else:
        click.echo(data)

@cli.command()
@click.pass_context
@click.option('--output', type=click.Path(dir_okay=False, writable=True), help='File path to write the output JSON.')
def last_three_releases(ctx: click.Context, output: Optional[click.Path]):
    """Fetch and print the last three release dates."""
    url = 'https://api.semanticscholar.org/datasets/v1/release'
    releases = make_request(url, ctx)
    write_output(releases[-3:], Path(output) if output else None)

@cli.command()
@click.pass_context
@click.option('--output', type=click.Path(dir_okay=False, writable=True), help='File path to write the output JSON.')
def latest_release(ctx: click.Context, output: Optional[click.Path]):
    """Fetch and print the latest release ID and details of the first dataset."""
    url = 'https://api.semanticscholar.org/datasets/v1/release/latest'
    latest = make_request(url, ctx)
    write_output(latest, Path(output) if output else None)

@cli.command()
@click.argument('dataset_name', type=str)
@click.pass_context
@click.option('--output', type=click.Path(dir_okay=False, writable=True), help='File path to write the output JSON.')
def dataset_info(ctx: click.Context, dataset_name: str, output: Optional[click.Path]):
    """Fetch and print info for a specific dataset in the latest release."""
    url = f'https://api.semanticscholar.org/datasets/v1/release/latest/dataset/{dataset_name}'
    dataset_info = make_request(url, ctx)
    write_output(dataset_info, Path(output) if output else None)

@cli.command()
@click.argument('dataset_name', type=str)
@click.pass_context
@click.option('--output', type=click.Path(dir_okay=False, writable=True), help='File path to write the output JSON.')
def incremental_updates(ctx: click.Context, dataset_name: str, output: Optional[click.Path]):
    """Fetch and print incremental updates for a specific dataset."""
    url = f'https://api.semanticscholar.org/datasets/v1/dataset/{dataset_name}/updates'
    updates = make_request(url, ctx)
    write_output(updates, Path(output) if output else None)

@cli.command()
@click.argument('release_id', type=str)
@click.argument('dataset_name', type=str)
@click.pass_context
@click.option('--output', type=click.Path(dir_okay=False, writable=True), help='File path to write the output JSON.')
def release_data(ctx: click.Context, release_id: str, dataset_name: str, output: Optional[click.Path]):
    """Fetch and print detailed release data for a specific dataset within a specified release."""
    url = f'https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset_name}'
    release_data = make_request(url, ctx)
    write_output(release_data, Path(output) if output else None)

if __name__ == '__main__':
    cli(obj={})
