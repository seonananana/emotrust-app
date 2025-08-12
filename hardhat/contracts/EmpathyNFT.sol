// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";

contract EmpathyNFT is ERC721URIStorage {
    uint256 private _nextTokenId;

    constructor() ERC721("EmpathyNFT", "EMO") {}

    // 테스트넷/로컬에서 누구나 민팅 가능 (필요하면 onlyOwner 붙이면 됨)
    function safeMint(address to, string memory tokenURI_) public returns (uint256) {
        uint256 tokenId = ++_nextTokenId;
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI_);
        return tokenId;
    }
}
