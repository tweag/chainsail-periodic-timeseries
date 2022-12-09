{ pkgs ? import <nixpkgs> { } }:
with pkgs;
let
  chainsailHelpersSrc = pkgs.fetchFromGitHub {
    owner = "tweag";
    repo = "chainsail-resources";
    rev = "5125a3277cb29c4cd5db38f33e899c5f050a1abd";
    sha256 = "adCRdWmWtrNjtB0vzeEFX6dRcCRQYc2JBip5I8rE+i4=";
  };
  chainsailHelpers = pkgs.lib.callPackageWith pkgs.python310Packages
    (chainsailHelpersSrc + "/examples/nix/chainsail_helpers.nix") { };
  pythonBundle =
    python310.withPackages (ps: with ps; [ matplotlib numpy chainsailHelpers ]);
in mkShell { buildInputs = [ pythonBundle ]; }
