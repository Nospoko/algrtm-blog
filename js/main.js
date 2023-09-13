/* global function */
import { initUtils } from "./utils.js";

document.addEventListener("DOMContentLoaded", () => {
  Global.themeInfo = {
    theme: `Redefine v${Global.theme_config.version}`,
    author: "EvanNotFound",
    repository: "https://github.com/EvanNotFound/hexo-theme-redefine",
  };

  Global.localStorageKey = "REDEFINE-THEME-STATUS";

  Global.styleStatus = {
    isExpandPageWidth: false,
    isDark: false,
    fontSizeLevel: 0,
    isOpenPageAside: true,
  };

  // print theme base info
  Global.printThemeInfo = () => {
    console.log('Welcome to the');
    console.log(' ________  ___       ________  ________  _________  _____ ______');
    console.log('|\\   __  \\|\\  \\     |\\   ____\\|\\   __  \\|\\___   ___\\\\   _ \\  _   \\');
    console.log('\\ \\  \\|\\  \\ \\  \\    \\ \\  \\___|\\ \\  \\|\\  \\|___ \\  \\_\\ \\  \\\\\\__\\ \\  \\');
    console.log(' \\ \\   __  \\ \\  \\    \\ \\  \\  __\\ \\   _  _\\   \\ \\  \\ \\ \\  \\\\|__| \\  \\');
    console.log('  \\ \\  \\ \\  \\ \\  \\____\\ \\  \\|\\  \\ \\  \\\\  \\|   \\ \\  \\ \\ \\  \\    \\ \\  \\');
    console.log('   \\ \\__\\ \\__\\ \\_______\\ \\_______\\ \\__\\\\ _\\    \\ \\__\\ \\ \\__\\    \\ \\__\\');
    console.log('    \\|__|\\|__|\\|_______|\\|_______|\\|__|\\|__|    \\|__|  \\|__|     \\|__|');
  };

  // set styleStatus to localStorage
  Global.setStyleStatus = () => {
    localStorage.setItem(
      Global.localStorageKey,
      JSON.stringify(Global.styleStatus),
    );
  };

  // get styleStatus from localStorage
  Global.getStyleStatus = () => {
    let temp = localStorage.getItem(Global.localStorageKey);
    if (temp) {
      temp = JSON.parse(temp);
      for (let key in Global.styleStatus) {
        Global.styleStatus[key] = temp[key];
      }
      return temp;
    } else {
      return null;
    }
  };

  Global.refresh = () => {
    initUtils();

    Global.initModeToggle();
    Global.initBackToTop();
    if (
      Global.theme_config.home_banner.subtitle.text.length !== 0 &&
      location.pathname === Global.hexo_config.root
    ) {
      Global.initTyped("subtitle");
    }

    if (Global.theme_config.navbar.search.enable === true) {
      Global.initLocalSearch();
    }

    if (Global.theme_config.articles.code_block.copy === true) {
      Global.initCopyCode();
    }

    if (Global.theme_config.articles.lazyload === true) {
      Global.initLazyLoad();
    }
  };

  Global.printThemeInfo();
  Global.refresh();
});
