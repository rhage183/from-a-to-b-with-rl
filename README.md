# From A to B with RL !

Hello !

## Setup

We work with the `rl` virtual environment, based on Python v3.11.9, so let us
first install that python version :

```
  pyenv install 3.11.9
```

If you don't have pyenv, please [install it first](https://github.com/pyenv/pyenv).
If you are on Windows, you can install [pyenv-win](https://pyenv-win.github.io/pyenv-win/)

You then have to create a virtual environment named `rl` :

```
  pyenv virtualenv 3.11.9 rl
````

Then, you can cd to the `from-a-to-b-with-rl` folder, and check if it is activated.

If the virtual environment is not activated automatically upon entering the folder you can run:

```
  pyenv local rl
````

### Packages

The `rl` virtual environment has a few dependencies, notably :

- [pytorch](https://pytorch.org/) for the RL learning
- [numpy](https://numpy.org/) to handle a few numerical outputs
- [gymnasium](https://gymnasium.farama.org/)
- [pygame](https://www.pygame.org/news)
- [moviepy](https://pypi.org/project/moviepy/) to save video from the agent interacting with the environment

You can then decide to install the package itself (XXX
Note, for now, nothing interesting is installed except from the dependencies XXX):

```
  pip install .
````

Or just decide to install the `requirements.txt` file :

```
  pip install -r requirements.txt
```

### Accessing a remote computer using SSH

Even if you are on windows, normally you _should_ have `ssh-keygen` and `ssh` available
as commands from the prompt.

First, let's create a SSH key using the [ed25519](https://en.wikipedia.org/wiki/EdDSA#Ed25519)
algorithm :

```
  ssh-keygen -t ed25519
```

It should normally ask you to enter a file name (see below typical output).
No bad choices here unless a file already exists, you can call it `rlkey`.

```
  Generating public/private ed25519 key pair.
  Enter file in which to save the key (/Users/indrianylionggo/.ssh/id_ed25519): mama
  Enter passphrase (empty for no passphrase):
```

It actually creates _two_ files, one being `<yourname>` (your _private_ key) and
one being  `<yourname>.pub`, both in your `~/.ssh/` folder. **Keep your private
key private !** and send the public key to Brice through Slack or anything.

Then, you can ask Brice to connect to his home laptop. I will not put my public
IP here since I am not crazy. It is available on the Slack channel. Let's say my
IP address is `XXX.XXX.XXX.XXX` :

```
  ssh lewagon@XXX.XXX.XXX.XXX
```

If Brice has done his job, your ssh authentication key will automatically be
recognized. However, your computer will not be sure of where it connects, so it
could throw a warning :

```
  The authenticity of host 'github.com (IP ADDRESS)' can't be established.
  ED25519 key fingerprint is SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU.
  Are you sure you want to continue connecting (yes/no)?
```

Type `yes`. You might be asked for either :

- the password for your private SSH key : only you know it.
- the password for the session `lewagon` : I have put it on our Slack channel.

Input them, then normally you should see :

```
  >> lewagon@pcbrice >
```

My machine is a Windows, so some Linux commands will not work (`touch`, maybe even `ls`). But

* python should work : `python --version`
* pyenv should work : `pyenv --help`
* git should work : `git --help`

Other interesting commands are `dir` (print folder contents) and `cd`. I'

#### On VSCode

Once you have checked that the connection works, you can try connecting _through_ VSCode. Click on the blue box (see below) and select `connect to host`, then ``

![img](vscode_ssh_screenshot.png)

Add a new ssh host and follow the instructions of the box that pops up, re-type the command :

```
  ssh lewagon@XXX.XXX.XXX.XXX
```

VSCode might ask you to edit the `.ssh` configuration file, I think it cannot
hurt. 

### Notes on GPU acceleration :

If your GPU is CUDA capable, you will have to adapt your `rl` environment. If you are on Windows, you can type :

```
  pip uninstall torch

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you are on Linux, you can do :

```
  pip uninstall torch

  pip3 install torch torchvision torchaudio
```
