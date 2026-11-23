"""Recalculate one workbook as SpreadsheetBench 2's ``evaluation/open_spreadsheet.py`` does.

Run by LibreOffice's Python (``program/python``, or the system ``python3`` that imports
``uno``), never imported: ``python -I lo_open_spreadsheet.py <program_dir> <soffice>
<profile_url> <pipe_name> <src> <dst>``. It starts a headless ``soffice`` on a private
profile and pipe (so concurrent runs never share one, unlike V2's fixed port 2002), opens
``src``, turns on iterative calculation (100 steps, 0.0001), recalculates everything and
stores the result to ``dst`` as "Calc MS Excel 2007 XML" (V2's ``process_file``, inside
``batch_open_files``' script), then stops ``soffice``. Prints one JSON line:
``{"ok": true}`` or ``{"ok": false, "error": ...}``.
"""
import json
import subprocess
import sys
import time


def main(argv):
    program_dir, soffice, profile_url, pipe, src, dst = argv[:6]
    # A tarball install's uno.py lives in its program directory (V2 puts it on PYTHONPATH,
    # open_spreadsheet.py:44-51; -I ignores that variable).
    sys.path.insert(0, program_dir)
    import uno
    from com.sun.star.beans import PropertyValue
    from com.sun.star.connection import NoConnectException

    proc = subprocess.Popen(
        [soffice, f"-env:UserInstallation={profile_url}", "--headless", "--invisible",
         "--norestore", "--nologo", "--nodefault", "--nofirststartwizard",
         f"--accept=pipe,name={pipe};urp;"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext(
            "com.sun.star.bridge.UnoUrlResolver", local)
        context = None
        # open_spreadsheet.py:121-133: wait for the service, up to its 30 s (and a little more
        # for a cold profile).
        deadline = time.monotonic() + 60
        while context is None:
            try:
                context = resolver.resolve(
                    f"uno:pipe,name={pipe};urp;StarOffice.ComponentContext")
            except NoConnectException:
                if proc.poll() is not None or time.monotonic() > deadline:
                    return {"ok": False, "error": "LibreOffice did not start"}
                time.sleep(0.5)
        desktop = context.ServiceManager.createInstanceWithContext(
            "com.sun.star.frame.Desktop", context)
        # open_spreadsheet.py process_file (inside batch_open_files' script, :202-222)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(src), "_blank", 0, ())
        if not doc:
            return {"ok": False, "error": "Failed to open document"}
        try:
            doc.IsIterationEnabled = True
            doc.IterationCount = 100
            doc.IterationEpsilon = 0.0001
            doc.calculateAll()
            save_props = [PropertyValue(Name="FilterName", Value="Calc MS Excel 2007 XML")]
            doc.storeToURL(uno.systemPathToFileUrl(dst), tuple(save_props))
        finally:
            doc.close(True)
        try:
            desktop.terminate()
        except Exception:  # noqa: BLE001 - the connection drops as it quits
            pass
        return {"ok": True}
    except Exception as e:  # noqa: BLE001 - reported, as process_file reports it
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:])), flush=True)
