{
  description = "Memory System";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
  
        pg_with_extension = pkgs.postgresql_17.withPackages (p: [
          p.pgvector 
        ]);
  
        start_db = pkgs.writeShellScriptBin "start-db" ''
          set -e
  	  export PGPORT="''${PGPORT:-5432}"
    	  export DB_DIR="$PWD/pg_data"
	  export PATH="${pg_with_extension}/bin:$PATH"
    
  	  if [ ! -f "$DB_DIR/PG_VERSION" ]; then
  	    echo "--- Initializing PostgreSQL in $DB_DIR ---"
  	    initdb -D "$DB_DIR" -U $USER
  	    # Socket and port
  	    echo "unix_socket_directories = '$DB_DIR'" >> "$DB_DIR/postgresql.conf"
	    echo "port = $PGPORT" >> "$DB_DIR/postgresql.conf"
	    # Extension
	    # echo "shared_preload_libraries = 'pgvector'" >> "$DB_DIR/postgresql.conf"
	  else
	    echo "--- Database already initialized in $DB_DIR ---"
	  fi
	  echo "--- Starting PostgreSQL ---"
	  pg_ctl -D "$DB_DIR" -l "$DB_DIR/logfile" start
	  echo "--- PostgreSQL is running! ---"
	  echo "To connect via Socket: psql -h '$DB_DIR' -U $USER postgres"
	  echo "To connect via TCP:    psql -h localhost -p $PGPORT -U $USER postgres"
	  echo "To stop:               pg_ctl -D $DB_DIR stop"
        '';

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pg_with_extension
	    start_db
	    pkgs.uv
	    pkgs.ninja
	    pkgs.ffmpeg
	  ];
        };
      });
}
