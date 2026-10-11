# Releasing contexteval

The release workflow (`.github/workflows/release.yml`) publishes to PyPI when you push a tag that starts with `v`. It runs the full test suite, checks that the tag matches the version in `pyproject.toml`, builds the sdist and wheel, runs `twine check`, and uploads through PyPI trusted publishing. The repository stores no PyPI token.

## One-time setup

Do these steps once, before the first release.

1. Sign in to https://pypi.org. Turn on two-factor authentication if the account does not have it.
2. Open https://pypi.org/manage/account/publishing/. The project does not exist yet, so add a *pending* publisher.
3. Under "Add a new pending publisher", select the **GitHub** tab and fill in these fields:
   - PyPI Project Name: `contexteval`
   - Owner: `aakarshan-coding`
   - Repository name: `context-evaluation-sdk`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
4. Click **Add**. The first successful publish creates the project and turns the pending publisher into a normal one. A pending publisher does not reserve the name, so publish soon after you add it.
5. On GitHub, open https://github.com/aakarshan-coding/context-evaluation-sdk/settings/environments and click **New environment**. Name it `pypi` and click **Configure environment**.
6. Optional: under **Deployment protection rules**, check **Required reviewers** and add yourself. Each release then waits for your approval before it uploads.
7. Optional: under **Deployment branches and tags**, choose **Selected branches and tags** and add the tag rule `v*`. Only version tags can then deploy to `pypi`.

## Each release

1. Set `version` in `pyproject.toml` to the new version, for example `0.2.0`.
2. In `CHANGELOG.md`, replace "unreleased" in that version's heading with the release date, and remove the "planned contents" note.
3. Commit both changes to `main` and push. Wait for the CI workflow to pass on that commit.
4. Tag the commit and push the tag:

   ```bash
   git checkout main
   git pull
   git tag -a v0.2.0 -m "contexteval 0.2.0"
   git push origin v0.2.0
   ```

5. Open the **Actions** tab on GitHub and watch the **Release** run. If you added required reviewers, approve the `pypi` deployment when it asks.
6. Check https://pypi.org/project/contexteval/ and install the new version in a clean virtual environment:

   ```bash
   pip install "contexteval[demo]==0.2.0"
   contexteval --help
   ```

If the tag does not match the `pyproject.toml` version, the build job fails before anything uploads. Delete the tag with `git push origin :refs/tags/v0.2.0` and `git tag -d v0.2.0`, fix the version, and tag again. PyPI never accepts the same version twice, so a version that did upload needs a new version number for any fix.
