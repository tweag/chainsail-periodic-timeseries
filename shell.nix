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
  # override to use GitHub as source: we need a newer feature (replica option for
  # concatenate-samples script) which is not on PyPi yet.
  chainsailHelpersFeature = chainsailHelpers.overridePythonAttrs (old: {
    src = fetchFromGitHub {
      owner = "tweag";
      repo = "chainsail-resources";
      rev = "5c0ae9e8426bb5bee9d0830f0617bb6747090b32";
      sha256 = "jF8qnTe2LNdUqLpQXnFrMrut9z128yYnAIF6e7LXT8M=";
    } + "/chainsail_helpers/";
    format = "pyproject";
    nativeBuildInputs = old.nativeBuildInputs or [ python310Packages.poetry ];
  });
  pythonBundle = python310.withPackages
    (ps: with ps; [ matplotlib numpy chainsailHelpersFeature ]);
in mkShell { buildInputs = [ pythonBundle ]; }
