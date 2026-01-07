import ppx

proj = ppx.find_project("PXD023456")
proj.download(proj.remote_files()) 
