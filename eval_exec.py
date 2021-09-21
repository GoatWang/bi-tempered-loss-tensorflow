import subprocess
for salt_percentage in [0.0, 0.05, 0.1, 0.2]:
    for t1 in [1.0, 0.6, 0.2]:
        for t2 in [1.0, 2.0, 4.0]:
            print("t1:", t1)
            print("t2:", t2)
            print("salt_percentage:", salt_percentage)
            subprocess.call('venv\Scripts\python eval.py ' + str(t1) + " " + str(t2) + " " + str(salt_percentage))
            print("=================")
