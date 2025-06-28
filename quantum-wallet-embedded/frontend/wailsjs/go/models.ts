export namespace main {
	
	export class Account {
	    address: string;
	    balance: string;
	    name: string;
	    isLocked: boolean;
	
	    static createFrom(source: any = {}) {
	        return new Account(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.address = source["address"];
	        this.balance = source["balance"];
	        this.name = source["name"];
	        this.isLocked = source["isLocked"];
	    }
	}
	export class MiningInfo {
	    isMining: boolean;
	    hashRate: string;
	    minerAddress: string;
	    blocksMined: number;
	
	    static createFrom(source: any = {}) {
	        return new MiningInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.isMining = source["isMining"];
	        this.hashRate = source["hashRate"];
	        this.minerAddress = source["minerAddress"];
	        this.blocksMined = source["blocksMined"];
	    }
	}
	export class NetworkInfo {
	    chainId: string;
	    networkName: string;
	    blockNumber: number;
	    peerCount: number;
	    syncing: boolean;
	    gasPrice: string;
	    difficulty: string;
	    hashRate: string;
	
	    static createFrom(source: any = {}) {
	        return new NetworkInfo(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.chainId = source["chainId"];
	        this.networkName = source["networkName"];
	        this.blockNumber = source["blockNumber"];
	        this.peerCount = source["peerCount"];
	        this.syncing = source["syncing"];
	        this.gasPrice = source["gasPrice"];
	        this.difficulty = source["difficulty"];
	        this.hashRate = source["hashRate"];
	    }
	}
	export class Transaction {
	    hash: string;
	    from: string;
	    to: string;
	    value: string;
	    gas: number;
	    gasPrice: string;
	    status: string;
	    blockNumber: number;
	    timestamp: number;
	    type: string;
	
	    static createFrom(source: any = {}) {
	        return new Transaction(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.hash = source["hash"];
	        this.from = source["from"];
	        this.to = source["to"];
	        this.value = source["value"];
	        this.gas = source["gas"];
	        this.gasPrice = source["gasPrice"];
	        this.status = source["status"];
	        this.blockNumber = source["blockNumber"];
	        this.timestamp = source["timestamp"];
	        this.type = source["type"];
	    }
	}
	export class WalletConfig {
	    dataDir: string;
	    network: string;
	    autoMining: boolean;
	    minerAddress: string;
	    externalMiner: boolean;
	    minerEndpoint: string;
	    httpPort: number;
	    wsPort: number;
	    enableLogging: boolean;
	    logLevel: string;
	
	    static createFrom(source: any = {}) {
	        return new WalletConfig(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.dataDir = source["dataDir"];
	        this.network = source["network"];
	        this.autoMining = source["autoMining"];
	        this.minerAddress = source["minerAddress"];
	        this.externalMiner = source["externalMiner"];
	        this.minerEndpoint = source["minerEndpoint"];
	        this.httpPort = source["httpPort"];
	        this.wsPort = source["wsPort"];
	        this.enableLogging = source["enableLogging"];
	        this.logLevel = source["logLevel"];
	    }
	}

}

