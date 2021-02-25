# Setup repository and environment on Piz-Daint
## 1. Access Daint
There are (at least?) two ways to access Daint:
1. `ssh <user>@ela.cscs.ch` from terminal and from there `ssh daint`. You can add ssh keys to both Ela and your personal computer to access the services passwordless (see https://user.cscs.ch/access/auth/#generating-ssh-keys). Note that you may need to add following lines to ~/.ssh/config (create the file if it does not exist yet)
```
Host ela.cscs.ch
    HostName ela.cscs.ch
    User <username>
    IdentityFile <path to key>
```
2. Via jupyter.cscs.ch

## 2. Set up environment
1. Start a fresh ssh session, login to daint
2. Set up ssh keys for github (https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)
3. `cd ~`
4. `git clone git@github.com:henrikhakansson/ska-sdc-2.git`
4. `source ska-sdc-2/environment/daint.bash`
5. Now you should be able to use notebooks found in repository, access from jupyter.cscs.ch