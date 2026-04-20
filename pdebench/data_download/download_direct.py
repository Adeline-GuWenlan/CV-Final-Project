from __future__ import annotations

import argparse
import http.client
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm

USER_AGENT = "pytorch/vision"
DOWNLOAD_TIMEOUT_SECONDS = 120
MAX_DOWNLOAD_RETRIES = 200
RETRY_DELAY_SECONDS = 5
MAX_STALLED_RETRIES = 5


def parse_metadata(pde_names):
    """
    This function parses the argument to filter the metadata of files that need to be downloaded.

    Args:
    pde_names: List containing the name of the PDE to be downloaded
    df      : The provided dataframe loaded from the csv file

    Options for pde_names:
    - Advection
    - Burgers
    - 1D_CFD
    - Diff-Sorp
    - 1D_ReacDiff
    - 2D_CFD
    - Darcy
    - 2D_ReacDiff
    - NS_Incom
    - SWE
    - 3D_CFD

    Returns:
    pde_df : Filtered dataframe containing metadata of files to be downloaded
    """

    meta_df = pd.read_csv("pdebench_data_urls.csv")

    # Ensure the pde_name is defined
    pde_list = [
        "advection",
        "burgers",
        "1d_cfd",
        "diff_sorp",
        "1d_reacdiff",
        "2d_cfd",
        "darcy",
        "2d_reacdiff",
        "ns_incom",
        "swe",
        "3d_cfd",
    ]

    assert all(name.lower() in pde_list for name in pde_names), "PDE name not defined."

    # Filter the files to be downloaded
    meta_df["PDE"] = meta_df["PDE"].str.lower()

    return meta_df[meta_df["PDE"].isin(pde_names)]


def resolve_root_folder(root_folder):
    """
    Expand a user-provided root folder and fail early with a helpful message
    when the destination is not writable.
    """

    resolved_root = Path(os.path.expandvars(os.path.expanduser(root_folder))).resolve()

    try:
        resolved_root.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create or write to root folder '{resolved_root}'. "
            "Please pass a writable path. If you used a shell variable such as "
            "'$proj_home/data', make sure that variable is defined before "
            "running the command."
        ) from exc

    return resolved_root


def _get_remote_file_info(url):
    """
    Resolve redirects and return the final URL plus remote file metadata.
    """

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urllib.request.urlopen(request) as response:
        content_length = response.headers.get("Content-Length")
        return {
            "url": response.geturl(),
            "size": int(content_length) if content_length is not None else None,
            "accept_ranges": "bytes" in response.headers.get("Accept-Ranges", "").lower(),
        }


def _local_file_size(path):
    return path.stat().st_size if path.exists() else 0


def download_url_with_resume(url, root, filename, md5=None):
    """
    Download a file and resume from an existing partial download when possible.
    Retries automatically when the remote side closes the connection early.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fpath = root / filename

    stalled_retries = 0
    previous_size = _local_file_size(fpath)
    last_error = None

    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        remote = _get_remote_file_info(url)
        total_size = remote["size"]
        existing_size = _local_file_size(fpath)

        if total_size is not None and existing_size == total_size:
            if md5 is None or check_integrity(fpath, md5):
                return

            tqdm.write(
                f"Checksum mismatch for {filename} even though the file size looks complete. "
                "Deleting the local copy and restarting that file from zero."
            )
            fpath.unlink()
            existing_size = 0

        elif total_size is not None and existing_size > total_size:
            tqdm.write(
                f"Local file {fpath} is larger than the remote file. "
                "Deleting the local copy and restarting that file from zero."
            )
            fpath.unlink()
            existing_size = 0

        request_headers = {"User-Agent": USER_AGENT}
        request_url = remote["url"]
        write_mode = "wb"

        if existing_size and remote["accept_ranges"]:
            request_headers["Range"] = f"bytes={existing_size}-"
            write_mode = "ab"
        elif existing_size:
            tqdm.write(
                f"The server does not advertise byte-range support for {filename}. "
                "Deleting the local partial file and restarting from zero."
            )
            fpath.unlink()
            existing_size = 0

        request = urllib.request.Request(request_url, headers=request_headers)
        transfer_error = None

        try:
            try:
                response = urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS)
            except (urllib.error.URLError, OSError):
                if request_url.startswith("https:"):
                    fallback_url = request_url.replace("https:", "http:", 1)
                    request = urllib.request.Request(fallback_url, headers=request_headers)
                    response = urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS)
                else:
                    raise

            with response:
                # Some servers ignore the Range header and send the full file. Restart in that case.
                if existing_size and getattr(response, "status", None) != 206:
                    existing_size = 0
                    write_mode = "wb"

                progress_total = total_size
                progress_desc = f"Downloading {filename}"

                with open(fpath, write_mode) as fh, tqdm(
                    total=progress_total,
                    initial=existing_size,
                    unit="B",
                    unit_scale=True,
                    desc=progress_desc,
                    leave=False,
                ) as pbar:
                    while True:
                        try:
                            chunk = response.read(1024 * 1024)
                        except http.client.IncompleteRead as exc:
                            chunk = exc.partial
                            transfer_error = exc
                            if chunk:
                                fh.write(chunk)
                                pbar.update(len(chunk))
                            break

                        if not chunk:
                            break

                        fh.write(chunk)
                        pbar.update(len(chunk))

        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            transfer_error = exc

        current_size = _local_file_size(fpath)

        if total_size is not None and current_size == total_size:
            if md5 is None or check_integrity(fpath, md5):
                return

            last_error = RuntimeError(
                f"Checksum mismatch after a complete-size download for {fpath}. "
                "The remote file may have changed or the transfer was corrupted."
            )
            if fpath.exists():
                fpath.unlink()
            current_size = 0
        elif total_size is not None and current_size < total_size:
            last_error = transfer_error or RuntimeError(
                f"Connection closed early for {fpath}: downloaded {current_size} bytes, "
                f"expected {total_size} bytes."
            )
        elif total_size is None and current_size and md5 and check_integrity(fpath, md5):
            return
        else:
            last_error = transfer_error or RuntimeError(f"Download failed for {fpath}.")

        if current_size > previous_size:
            stalled_retries = 0
        else:
            stalled_retries += 1
        previous_size = current_size

        if attempt >= MAX_DOWNLOAD_RETRIES or stalled_retries >= MAX_STALLED_RETRIES:
            raise RuntimeError(
                f"File not found or corrupted after repeated retry attempts: {fpath} "
                f"(downloaded {current_size} bytes"
                + (f", expected {total_size} bytes" if total_size is not None else "")
                + f"). Last error: {last_error}"
            )

        tqdm.write(
            f"Download interrupted for {filename}. Retrying from byte {current_size}"
            + (f" of {total_size}" if total_size is not None else "")
            + f" in {RETRY_DELAY_SECONDS} seconds "
            + f"(attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES})."
        )
        time.sleep(RETRY_DELAY_SECONDS)


def download_data(root_folder, pde_name):
    """ "
    Download data splits specific to a given PDE.

    Args:
    root_folder: The root folder where the data will be downloaded
    pde_name   : The name of the PDE for which the data to be downloaded
    """

    # print(f"Downloading data for {pde_name} ...")

    # Load and parse metadata csv file
    pde_df = parse_metadata(pde_name)
    root_path = resolve_root_folder(root_folder)

    # Iterate filtered dataframe and download the files
    for _, row in tqdm(pde_df.iterrows(), total=pde_df.shape[0]):
        file_path = root_path / row["Path"]
        download_url_with_resume(row["URL"], file_path, row["Filename"], md5=row["MD5"])


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Download Script",
        description="Helper script to download the PDEBench datasets",
        epilog="",
    )

    arg_parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder where the data will be downloaded",
    )
    arg_parser.add_argument(
        "--pde_name",
        action="append",
        help="Name of the PDE dataset to download. You can use this flag multiple times to download multiple datasets",
    )

    args = arg_parser.parse_args()

    download_data(args.root_folder, args.pde_name)
