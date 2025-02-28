# Semantic Versioning Guide

This project uses [Semantic Versioning](https://semver.org/) for version management and [Python Semantic Release](https://python-semantic-release.readthedocs.io/) for automating releases.

## Commit Message Format

To trigger automatic version bumps, your commit messages should follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types that trigger version changes:

- `fix:` - Patches a bug (PATCH version bump)
- `feat:` - Adds a new feature (MINOR version bump)
- `BREAKING CHANGE:` - Introduces a breaking API change (MAJOR version bump)
  - This can be in the footer of any commit type

### Other common types (no version bump):

- `docs:` - Documentation only changes
- `style:` - Changes that do not affect the meaning of the code
- `refactor:` - Code change that neither fixes a bug nor adds a feature
- `perf:` - Code change that improves performance
- `test:` - Adding missing tests or correcting existing tests
- `chore:` - Changes to the build process or auxiliary tools

### Examples:

```
feat(vectorstore): add support for multi-document retrieval

This adds the ability to retrieve multiple documents at once from the vector store.
```

```
fix(embeddings): correct dimension calculation

The embedding dimension was incorrectly calculated, causing issues with vector storage.
```

```
feat(api): add new endpoint for document search

BREAKING CHANGE: The response format for search results has changed from a list to a dictionary with metadata.
```

## Manual Release

If you need to trigger a release manually:

```bash
semantic-release version
semantic-release publish
```

## Checking the Next Version

To check what the next version would be based on your commits:

```bash
semantic-release version --print
```

## Viewing the Changelog

The changelog is automatically generated and updated in the CHANGELOG.md file.
