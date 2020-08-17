### Submodules

[Submodules basic](https://chrisjean.com/git-submodules-adding-using-removing-and-updating/)


# Chapter 1
![](master/chapter1git.png)

# Chapter 2
![](master/chapter2git.png)

# Chapter 3
![](master/chapter3git.png)

# Chapter 4
![](master/chapter4git.png)

### Fish configuration
``` shell
function fish_prompt
    set_color normal
    # https://stackoverflow.com/questions/24581793/ps1-prompt-in-fish-friendly-interactive-shell-show-git-branch
    set -l git_branch (git branch 2>/dev/null | sed -n '/\* /s///p')
    echo -n (whoami)'@'(hostname)':'
    set_color $fish_color_cwd
    echo -n (prompt_pwd)
    set_color normal
    echo -n '{'
    set_color purple
    echo -n "$git_branch"
    set_color normal
    echo -n '}'
    echo -n ' $ '
end
```