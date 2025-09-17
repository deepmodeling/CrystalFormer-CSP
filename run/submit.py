#!/usr/bin/env python
import sys , os 

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action='store_true', help="Run or not")
    parser.add_argument("--waitfor", type=int, help="wait for this job for finish")
    input = parser.parse_args()


    #this import might overwrite the above default parameters 
    #########################################################
    import socket, getpass
    machinename = socket.gethostname()
    username = getpass.getuser()
    print ('\n', username, 'on', machinename, '\n')
    if 'ip-10-0-0-26' in machinename:
        from config.aws import * 
    elif 'ln01' in machinename:
        from config.ln01 import * 
    elif 'bright90' in machinename:
        from config.bright90 import * 
    elif 'slurm-client' in machinename:
        from config.pmax import *
    else:
        print ('where am I ?', machinename)
        sys.exit(1)
    #########################################################

    from pygit2 import Repository
    head = Repository('.').head
    branch = head.shorthand + "-" + head.target.raw.hex()[:5]

    resfolder = os.path.join(resfolder, branch) + '/'

    jobdir='../jobs/' + nickname + '/'
    jobdir = os.path.join(jobdir, branch) + '/'

    cmd = ['mkdir', '-p', jobdir]
    subprocess.check_call(cmd)

    cmd = ['mkdir', '-p', resfolder]
    subprocess.check_call(cmd)
    
    if True:
                args = {
                        'epochs':epochs, 
                        'batchsize':batchsize, 
                        'folder':resfolder,
                        'reward': reward, 
                        'elements': elements, 
                        'spacegroup': spacegroup, 
                        'mlff_model': mlff_model, 
                        'restore_path': restore_path, 
                        'convex_path': convex_path, 
                        'mlff_path': mlff_path, 
                        'beta':beta
                        }

                logname = jobdir 
                for arg, value in args.items():
                    if isinstance(value, bool):
                        logname += ("%s_" % arg if value else "")
                    elif not ('_path' in arg or 'folder' in arg):
                        if '_' in arg:
                            arg = "".join([s[0] for s in arg.split('_')])
                        logname += "%s%s_" % (arg, value)
                logname = logname[:-1] + '.log'

                jobname = os.path.basename(os.path.dirname(logname))

                jobid = submitJob(prog,args,jobname,logname,run=input.run, wait=input.waitfor)
