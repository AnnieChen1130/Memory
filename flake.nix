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

	    start_minio = pkgs.writeShellScriptBin "start-minio" ''
          set -e
          export MINIO_DATA_DIR="$PWD/minio_data"
          export MINIO_PORT="''${MINIO_PORT:-9000}"
          export MINIO_CONSOLE_PORT="''${MINIO_CONSOLE_PORT:-9001}"
          export MINIO_ACCESS_KEY="''${MINIO_ACCESS_KEY:-minioadmin}"
          export MINIO_SECRET_KEY="''${MINIO_SECRET_KEY:-minioadmin}"

          mkdir -p "$MINIO_DATA_DIR"

          echo "--- Starting MinIO S3 Storage ---"
          echo "Data directory:    $MINIO_DATA_DIR"
          echo "API Port:          $MINIO_PORT"
          echo "Console Port:      $MINIO_CONSOLE_PORT"
          echo "Access Key (User): $MINIO_ACCESS_KEY"
          # echo "Secret Key (Pass): $MINIO_SECRET_KEY"
          echo "Web UI:            http://127.0.0.1:$MINIO_CONSOLE_PORT"
          echo "------------------------------------"
          echo "To stop MinIO, press Ctrl+C in this terminal."

          minio server --address ":$MINIO_PORT" --console-address ":$MINIO_CONSOLE_PORT" "$MINIO_DATA_DIR"
        '';

      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pg_with_extension
            start_db
			pkgs.minio
			start_minio
	        pkgs.uv
            pkgs.ninja
            pkgs.ffmpeg
	  ];
        };
      });
}
