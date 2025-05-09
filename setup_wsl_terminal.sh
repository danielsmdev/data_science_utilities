#!/bin/bash

# Instala ZSH si no está instalado
if ! command -v zsh &> /dev/null; then
    echo "🔧 Instalando zsh..."
    sudo apt update && sudo apt install zsh -y
fi

# Instalar Oh My Zsh si no está instalado
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "💡 Instalando Oh My Zsh..."
    RUNZSH=no CHSH=no sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi

# Instalar tema Powerlevel10k
echo "🎨 Instalando Powerlevel10k..."
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git \
  ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

# Instalar plugins útiles
echo "✨ Instalando plugins zsh-autosuggestions y zsh-syntax-highlighting..."
git clone https://github.com/zsh-users/zsh-autosuggestions \
  ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

git clone https://github.com/zsh-users/zsh-syntax-highlighting \
  ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Modificar .zshrc
echo "🛠 Configurando .zshrc..."

sed -i 's/^ZSH_THEME=.*/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc
sed -i 's/^plugins=.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# Añadir configuración si no existe
if ! grep -q "powerlevel10k/powerlevel10k" ~/.zshrc; then
    echo 'ZSH_THEME="powerlevel10k/powerlevel10k"' >> ~/.zshrc
fi

# Cambiar shell por defecto a zsh si aún no lo es
if [ "$SHELL" != "$(which zsh)" ]; then
    echo "✅ Cambiando el shell por defecto a zsh..."
    chsh -s $(which zsh)
fi

echo -e "\n🚀 ¡Listo! Reinicia tu terminal o ejecuta 'zsh' para empezar a usar tu nueva shell con Git y Powerlevel10k."
